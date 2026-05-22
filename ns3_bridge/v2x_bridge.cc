/*
 * v2x_bridge.cc — Standalone V2X Communication Simulator
 *
 * Channel model: 3GPP TR 37.885 V2V highway
 *   Path loss  : Table A.1-2 LOS (fc=5.9 GHz default)
 *   Shadowing  : Log-normal, sigma=3 dB
 *   Fast fading: Rayleigh, power ~ Exp(1)
 *   PDR mapping: Sigmoid BLER curve per MCS (3GPP TR 38.901 calibrated)
 *   V2V pairing: NEAREST-NEIGHBOUR (not circular) — critical for correct SINR
 *
 * Interface: ZMQ REP socket, JSON over TCP port 5556
 *   Request : { "vehicles":[[x,y],...], "subchannels":[...], "powers_dBm":[...] }
 *   Reply   : { "sinr_lin":[...], "sinr_dB":[...], "pdr":[...] }
 *
 * Compile (from ~/v2x_thesis/ns3_bridge/):
 *   g++ -std=c++17 -O2 -o v2x_bridge v2x_bridge.cc -I/usr/include -lzmq
 */

#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <iostream>

using json = nlohmann::json;

// ── RNG: thread-local Mersenne Twister ────────────────────────────────
// Replace std::random_device{}() with a fixed seed (e.g. 42) for
// reproducible debugging runs.
static std::mt19937& rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

// ── 3GPP TR 37.885 Table A.1-2: V2V Highway LOS Path Loss ─────────────
// PL = 32.4 + 20*log10(fc_ghz) + 20*log10(d_m)  [dB]
// d_m    : TX-RX distance [m], clamped to 1.0 m minimum
// fc_ghz : carrier frequency [GHz]
//   5.9  GHz : EU ITS-G5 / C-V2X PC5 (default)
//   5.855 GHz: US DSRC
//   3.5  GHz : sub-6 GHz NR (different propagation model needed)
double path_loss(double d_m, double fc_ghz = 5.9) {
    if (d_m < 1.0) d_m = 1.0;
    return 32.4 + 20.0*std::log10(fc_ghz) + 20.0*std::log10(d_m);
}

// ── Log-normal Shadowing ───────────────────────────────────────────────
// sigma_db: standard deviation [dB]
//   3.0 dB : TR 37.885 highway LOS (default)
//   4.0 dB : urban NLOS
//   5.0 dB : conservative mixed
double shadowing(double sigma_db = 3.0) {
    std::normal_distribution<double> nd(0.0, sigma_db);
    return nd(rng());
}

// ── Rayleigh Fast Fading ───────────────────────────────────────────────
// Power ~ Exp(1) = Nakagami-m with m=1.
double fast_fading() {
    std::exponential_distribution<double> ed(1.0);
    return ed(rng());
}

// ── SINR for all N vehicles in one TTI ───────────────────────────────
// noise_dbm: thermal noise floor
//   -114 dBm : 10 MHz BW, T=290K, NF=9 dB (default)
//   -117 dBm : 5 MHz BW
//   -111 dBm : 20 MHz BW
std::vector<double> compute_sinr(
    const std::vector<std::pair<double,double>>& pos,
    const std::vector<int>&    ch,
    const std::vector<double>& pwr_dbm,
    int N, double noise_dbm=-114.0, double fc_ghz=5.9)
{
    double noise_lin = std::pow(10.0, noise_dbm/10.0);
    std::vector<double> sinr(N, 0.0);

    for (int i = 0; i < N; ++i) {
        double Ptx = std::pow(10.0, pwr_dbm[i]/10.0);

        // ── NEAREST-NEIGHBOUR pairing ─────────────────────────────────
        // Vehicle i transmits to the closest other vehicle.
        // This ensures the intended receiver is NEVER farther than the
        // closest interferer, giving geometrically correct SINR > 0 dB
        // under light load conditions.
        //
        // WHY NOT circular (i+1)%N:
        // SUMO orders vehicles by spawn time. Vehicle 0 (near entry) paired
        // with vehicle N-1 (far end of road) = 3 km distance, while close
        // vehicles interfere at 50 m → SINR = -20 dB by geometry alone.
        // This was the root cause of M0 SINR = -11 dB in early runs.
        int rx = -1;
        double d_min = 1e9;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            double dxj = pos[i].first  - pos[j].first;
            double dyj = pos[i].second - pos[j].second;
            double dj  = std::sqrt(dxj*dxj + dyj*dyj);
            if (dj < d_min) { d_min = dj; rx = j; }
        }
        double d = std::max(1.0, d_min);

        double pl_db = path_loss(d, fc_ghz);
        double sh_db = shadowing(3.0);
        double ff    = fast_fading();
        double S     = Ptx * std::pow(10.0,-(pl_db+sh_db)/10.0) * ff;

        // Co-channel interference from all other vehicles on same subchannel
        double I = 0.0;
        for (int k = 0; k < N; ++k) {
            if (k==i || ch[k]!=ch[i]) continue;
            double Pk  = std::pow(10.0, pwr_dbm[k]/10.0);
            double dxk = pos[k].first - pos[rx].first;
            double dyk = pos[k].second - pos[rx].second;
            double dk  = std::max(1.0, std::sqrt(dxk*dxk+dyk*dyk));
            I += Pk * std::pow(10.0,-path_loss(dk,fc_ghz)/10.0) * fast_fading();
        }
        sinr[i] = S / (I + noise_lin);
    }
    return sinr;
}

// ── 3GPP NR-V2X BLER-based SINR→PDR ──────────────────────────────────
// Sigmoid BLER curve per MCS, calibrated from TR 38.901 / TS 38.214:
//   Index 0: QPSK  CR=1/8  thr=-3.0 dB  steep=1.5  (most robust)
//   Index 1: QPSK  CR=1/2  thr= 1.5 dB  steep=1.8  (safety msgs, default)
//   Index 2: 16QAM CR=1/2  thr= 8.5 dB  steep=2.0  (balanced)
//   Index 3: 64QAM CR=3/4  thr=18.0 dB  steep=2.5  (high throughput)
// packet_size_bytes:
//   300  : V2V safety message / CAM / DENM (default)
//   1500 : V2I data / HD map update
struct MCSParams { double thr; double steep; };
static const MCSParams MCS[] = {{-3.0,1.5},{1.5,1.8},{8.5,2.0},{18.0,2.5}};
static const int N_MCS = 4;

double sinr_to_pdr(double sinr_lin, int mcs=1, int pkt_bytes=300) {
    if (mcs < 0) mcs=0; if (mcs>=N_MCS) mcs=N_MCS-1;
    double sdB   = 10.0*std::log10(sinr_lin+1e-12);
    double bler  = 1.0/(1.0+std::exp(MCS[mcs].steep*(sdB-MCS[mcs].thr)));
    int    nblk  = (int)std::ceil(pkt_bytes*8.0/8448.0);
    if (nblk<1) nblk=1;
    return std::max(0.0, std::min(1.0, std::pow(1.0-bler, nblk)));
}

// ── Main: ZMQ REP loop ────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    uint16_t port=5556; double fc_ghz=5.9;
    for (int i=1;i<argc;++i) {
        std::string a(argv[i]);
        if (a.rfind("--port=",0)==0)  port   = std::stoi(a.substr(7));
        if (a.rfind("--fcGHz=",0)==0) fc_ghz = std::stod(a.substr(8));
    }
    std::cout<<"[V2X-Bridge] ZMQ REP tcp://*:"<<port
             <<"  fc="<<fc_ghz<<" GHz\n";

    zmq::context_t ctx(1);
    zmq::socket_t  sock(ctx, zmq::socket_type::rep);
    sock.bind("tcp://*:"+std::to_string(port));
    std::cout<<"[V2X-Bridge] Ready.\n";

    while (true) {
        zmq::message_t req_msg;
        auto rc = sock.recv(req_msg, zmq::recv_flags::none);
        if (!rc) continue;
        std::string req_str(static_cast<char*>(req_msg.data()),req_msg.size());
        json req, rep;
        try { req=json::parse(req_str); }
        catch(const std::exception& e) {
            rep["error"]=e.what();
            std::string s=rep.dump();
            sock.send(zmq::buffer(s),zmq::send_flags::none); continue;
        }
        int N=(int)req["vehicles"].size();
        std::vector<std::pair<double,double>> pos(N);
        std::vector<int> ch(N); std::vector<double> pwr(N);
        for (int i=0;i<N;++i) {
            pos[i]={req["vehicles"][i][0],req["vehicles"][i][1]};
            ch[i]=req["subchannels"][i]; pwr[i]=req["powers_dBm"][i];
        }
        auto sinr=compute_sinr(pos,ch,pwr,N,-114.0,fc_ghz);
        rep["sinr_lin"]=json::array();
        rep["sinr_dB"] =json::array();
        rep["pdr"]     =json::array();
        for (int i=0;i<N;++i) {
            double sl=sinr[i];
            rep["sinr_lin"].push_back(sl);
            rep["sinr_dB"].push_back(10.0*std::log10(sl+1e-12));
            rep["pdr"].push_back(sinr_to_pdr(sl,1,300));
        }
        std::string rs=rep.dump();
        sock.send(zmq::buffer(rs),zmq::send_flags::none);
    }
    return 0;
}

// jet_eec_fluct.cc
//
// Minimal runnable code for:
//   - Pythia8 event generation
//   - FastJet anti-kt jet clustering
//   - jet selection
//   - single-jet EEC calculation
//   - mean / variance / relative fluctuation
//   - covariance / correlation matrix
//   - save per-jet EEC vectors to ROOT
//
// Compile example:
// g++ -O2 -std=c++17 jet_eec_fluct.cc -o jet_eec_fluct \
//   $(pythia8-config --cxxflags --libs) \
//   $(fastjet-config --cxxflags --libs) \
//   $(root-config --cflags --libs)
//
// Run:
// ./jet_eec_fluct
//
// Notes:
// 1) Default setup: pp, sqrt(s)=13 TeV, anti-kt R=0.4
// 2) Jet selection: 100 < pT_jet < 120 GeV, |eta_jet| < 1.0
// 3) EEC weight: w_ij = pT_i * pT_j / pT_jet^2
// 4) Angular variable: DeltaR_ij in log bins from 1e-3 to R_jet
//
/*
g++ -O2 -std=c++17 jet_eec_fluct.cc -o jet_eec_fluct \
  $(pythia8-config --cxxflags --libs) \
  $(fastjet-config --cxxflags --libs) \
  $(root-config --cflags --libs)

./jet_eec_fluct
*/


#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>

#include "Pythia8/Pythia.h"
#include "fastjet/ClusterSequence.hh"

#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TNamed.h"
#include "TParameter.h"

using namespace std;
using namespace Pythia8;
using namespace fastjet;

struct Config
{
    // -------- physics setup --------
    int    nEvent       = 100000;
    double eCM          = 5020.0;   // GeV
    int    seed         = 12345;

    // -------- jet setup --------
    double jetR         = 0.4;
    double jetPtMin     = 30.0;     // GeV
    double jetPtMax     = 50.0;     // GeV
    double jetEtaMax    = 5.0;
    bool   chargedOnly  = false;     // false = all visible final-state particles

    // -------- EEC setup --------
    int    nRBins       = 30;
    double rMin         = 1.0e-3;    // lower edge of DeltaR
    bool   normalizeByBinWidth = true;

    // -------- output --------
    string outFileName  = "jet_eec_fluct.root";
};

double deltaPhi(double phi1, double phi2)
{
    double dphi = phi1 - phi2;
    while (dphi >  M_PI) dphi -= 2.0 * M_PI;
    while (dphi < -M_PI) dphi += 2.0 * M_PI;
    return dphi;
}

double deltaR(double eta1, double phi1, double eta2, double phi2)
{
    const double deta = eta1 - eta2;
    const double dphi = deltaPhi(phi1, phi2);
    return std::sqrt(deta * deta + dphi * dphi);
}

bool isInvisibleNeutrino(int pdg)
{
    int apdg = std::abs(pdg);
    return (apdg == 12 || apdg == 14 || apdg == 16 || apdg == 18);
}

vector<double> makeLogBins(double xmin, double xmax, int nbins)
{
    vector<double> edges(nbins + 1, 0.0);
    double logMin = std::log(xmin);
    double logMax = std::log(xmax);
    double step   = (logMax - logMin) / nbins;
    for (int i = 0; i <= nbins; ++i)
    {
        edges[i] = std::exp(logMin + i * step);
    }
    return edges;
}

int findBin(double x, const vector<double>& edges)
{
    if (x < edges.front() || x >= edges.back()) return -1;
    auto it = std::upper_bound(edges.begin(), edges.end(), x);
    int idx = int(it - edges.begin()) - 1;
    if (idx < 0 || idx >= (int)edges.size() - 1) return -1;
    return idx;
}

vector<double> computeJetEEC(const fastjet::PseudoJet& jet,
                             const vector<double>& rEdges,
                             bool normalizeByBinWidth)
{
    const auto constituents = jet.constituents();
    const int nBins = (int)rEdges.size() - 1;
    vector<double> eec(nBins, 0.0);

    if (constituents.size() < 2) return eec;

    const double jetPt = jet.pt();
    if (jetPt <= 0.0) return eec;

    for (size_t i = 0; i < constituents.size(); ++i)
    {
        const double pti  = constituents[i].pt();
        const double etai = constituents[i].eta();
        const double phii = constituents[i].phi_std();

        for (size_t j = i + 1; j < constituents.size(); ++j)
        {
            const double ptj  = constituents[j].pt();
            const double etaj = constituents[j].eta();
            const double phij = constituents[j].phi_std();

            const double dr = deltaR(etai, phii, etaj, phij);
            int ibin = findBin(dr, rEdges);
            if (ibin < 0) continue;

            double w = (pti * ptj) / (jetPt * jetPt);

            if (normalizeByBinWidth)
            {
                const double bw = rEdges[ibin + 1] - rEdges[ibin];
                w /= bw;
            }

            eec[ibin] += w;
        }
    }

    return eec;
}

int main()
{
    Config cfg;

    // ------------------------------
    // Print config
    // ------------------------------
    cout << "\n========== jet EEC fluctuation job ==========\n";
    cout << "nEvent              = " << cfg.nEvent << "\n";
    cout << "sqrt(s)             = " << cfg.eCM << " GeV\n";
    cout << "seed                = " << cfg.seed << "\n";
    cout << "jetR                = " << cfg.jetR << "\n";
    cout << "jetPtMin            = " << cfg.jetPtMin << "\n";
    cout << "jetPtMax            = " << cfg.jetPtMax << "\n";
    cout << "jetEtaMax           = " << cfg.jetEtaMax << "\n";
    cout << "chargedOnly         = " << (cfg.chargedOnly ? "true" : "false") << "\n";
    cout << "nRBins              = " << cfg.nRBins << "\n";
    cout << "rMin                = " << cfg.rMin << "\n";
    cout << "normalizeByBinWidth = " << (cfg.normalizeByBinWidth ? "true" : "false") << "\n";
    cout << "outFile             = " << cfg.outFileName << "\n";
    cout << "============================================\n\n";

    // ------------------------------
    // EEC binning
    // ------------------------------
    vector<double> rEdges = makeLogBins(cfg.rMin, cfg.jetR, cfg.nRBins);
    vector<double> rCenters(cfg.nRBins, 0.0);
    for (int i = 0; i < cfg.nRBins; ++i)
    {
        rCenters[i] = std::sqrt(rEdges[i] * rEdges[i + 1]); // geometric center
    }

    // ------------------------------
    // Pythia setup
    // ------------------------------
    Pythia pythia;
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = " + std::to_string(cfg.eCM));

    // Hard QCD inclusive
    pythia.readString("HardQCD:all = on");

    // seed
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + std::to_string(cfg.seed));

    pythia.readString("PhaseSpace:pTHatMin = 20.");

    pythia.init();

    // ------------------------------
    // FastJet setup
    // ------------------------------
    JetDefinition jetDef(antikt_algorithm, cfg.jetR);

    // ------------------------------
    // Output ROOT
    // ------------------------------
    TFile* fout = TFile::Open(cfg.outFileName.c_str(), "RECREATE");
    if (!fout || fout->IsZombie())
    {
        cerr << "ERROR: cannot create output ROOT file: " << cfg.outFileName << endl;
        return 1;
    }

    TH1D* hJetPt   = new TH1D("hJetPt",   ";p_{T}^{jet} [GeV];Jets", 100, 0.0, 200.0);
    TH1D* hJetEta  = new TH1D("hJetEta",  ";#eta^{jet};Jets",        60, -3.0, 3.0);
    TH1D* hNConst  = new TH1D("hNConst",  ";N constituents;Jets",   100, 0.0, 100.0);
    TH1D* hMeanEEC = new TH1D("hMeanEEC", ";#DeltaR;#LT EEC #GT", cfg.nRBins, rEdges.data());
    TH1D* hVarEEC  = new TH1D("hVarEEC",  ";#DeltaR;Var(EEC)",    cfg.nRBins, rEdges.data());
    TH1D* hRelFluc = new TH1D("hRelFluc", ";#DeltaR;#sigma/#LT EEC #GT", cfg.nRBins, rEdges.data());

    TH2D* hCov = new TH2D("hCov", ";#DeltaR_{a};#DeltaR_{b};Cov",
                          cfg.nRBins, rEdges.data(), cfg.nRBins, rEdges.data());
    TH2D* hCorr = new TH2D("hCorr", ";#DeltaR_{a};#DeltaR_{b};Corr",
                           cfg.nRBins, rEdges.data(), cfg.nRBins, rEdges.data());

    // per-jet tree
    TTree* tree = new TTree("JetTree", "Per-jet EEC vectors");

    int    t_event = -1;
    double t_jet_pt = 0.0, t_jet_eta = 0.0, t_jet_phi = 0.0, t_jet_m = 0.0;
    int    t_nconst = 0;
    vector<float> t_eec;
    vector<float> t_rcenter;

    tree->Branch("event",   &t_event);
    tree->Branch("jet_pt",  &t_jet_pt);
    tree->Branch("jet_eta", &t_jet_eta);
    tree->Branch("jet_phi", &t_jet_phi);
    tree->Branch("jet_m",   &t_jet_m);
    tree->Branch("nconst",  &t_nconst);
    tree->Branch("eec",     &t_eec);
    tree->Branch("rcenter", &t_rcenter);

    // ------------------------------
    // Accumulators
    // ------------------------------
    long long nAcceptedJets = 0;
    vector<double> sumEEC(cfg.nRBins, 0.0);
    vector<double> sumEEC2(cfg.nRBins, 0.0);
    vector<vector<double>> sumCross(cfg.nRBins, vector<double>(cfg.nRBins, 0.0));

    // ------------------------------
    // Event loop
    // ------------------------------
    for (int iEvent = 0; iEvent < cfg.nEvent; ++iEvent)
    {
        if (!pythia.next()) continue;

        vector<PseudoJet> fjInputs;
        fjInputs.reserve(pythia.event.size());

        for (int i = 0; i < pythia.event.size(); ++i)
        {
            const Particle& p = pythia.event[i];

            if (!p.isFinal()) continue;
            if (!p.isVisible()) continue;
            if (isInvisibleNeutrino(p.id())) continue;
            if (cfg.chargedOnly && !p.isCharged()) continue;

            PseudoJet pj(p.px(), p.py(), p.pz(), p.e());
            pj.set_user_index(i);
            fjInputs.push_back(pj);
        }

        if (fjInputs.empty()) continue;

        ClusterSequence cs(fjInputs, jetDef);
        vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(0.0));

        for (const auto& jet : jets)
        {
            if (jet.pt() < cfg.jetPtMin || jet.pt() >= cfg.jetPtMax) continue;
            if (std::abs(jet.eta()) > cfg.jetEtaMax) continue;

            vector<double> eec = computeJetEEC(jet, rEdges, cfg.normalizeByBinWidth);

            // tree fill
            t_event   = iEvent;
            t_jet_pt  = jet.pt();
            t_jet_eta = jet.eta();
            t_jet_phi = jet.phi_std();
            t_jet_m   = jet.m();
            t_nconst  = (int)jet.constituents().size();

            t_eec.clear();
            t_rcenter.clear();
            t_eec.reserve(cfg.nRBins);
            t_rcenter.reserve(cfg.nRBins);

            for (int b = 0; b < cfg.nRBins; ++b)
            {
                t_eec.push_back((float)eec[b]);
                t_rcenter.push_back((float)rCenters[b]);
            }
            tree->Fill();

            // histograms
            hJetPt->Fill(jet.pt());
            hJetEta->Fill(jet.eta());
            hNConst->Fill(t_nconst);

            // accumulators
            for (int a = 0; a < cfg.nRBins; ++a)
            {
                sumEEC[a]  += eec[a];
                sumEEC2[a] += eec[a] * eec[a];

                for (int b = 0; b < cfg.nRBins; ++b)
                {
                    sumCross[a][b] += eec[a] * eec[b];
                }
            }

            ++nAcceptedJets;
        }

        if ((iEvent + 1) % 5000 == 0)
        {
            cout << "Processed " << (iEvent + 1)
                 << " / " << cfg.nEvent
                 << " events, accepted jets = " << nAcceptedJets << endl;
        }
    }

    // ------------------------------
    // Final statistics
    // ------------------------------
    if (nAcceptedJets == 0)
    {
        cerr << "ERROR: no accepted jets. Try increasing nEvent or loosening jet cuts." << endl;
        fout->Close();
        return 1;
    }

    vector<double> mean(cfg.nRBins, 0.0);
    vector<double> var(cfg.nRBins, 0.0);
    vector<double> sigma(cfg.nRBins, 0.0);

    for (int a = 0; a < cfg.nRBins; ++a)
    {
        mean[a] = sumEEC[a] / nAcceptedJets;
        var[a]  = sumEEC2[a] / nAcceptedJets - mean[a] * mean[a];
        if (var[a] < 0.0 && std::abs(var[a]) < 1e-15) var[a] = 0.0;
        sigma[a] = std::sqrt(std::max(0.0, var[a]));

        hMeanEEC->SetBinContent(a + 1, mean[a]);
        hVarEEC->SetBinContent(a + 1, var[a]);
        hRelFluc->SetBinContent(a + 1, (mean[a] > 0.0 ? sigma[a] / mean[a] : 0.0));
    }

    for (int a = 0; a < cfg.nRBins; ++a)
    {
        for (int b = 0; b < cfg.nRBins; ++b)
        {
            double cov = sumCross[a][b] / nAcceptedJets - mean[a] * mean[b];
            if (std::abs(cov) < 1e-15) cov = 0.0;

            double corr = 0.0;
            if (sigma[a] > 0.0 && sigma[b] > 0.0)
            {
                corr = cov / (sigma[a] * sigma[b]);
            }

            hCov->SetBinContent(a + 1, b + 1, cov);
            hCorr->SetBinContent(a + 1, b + 1, corr);
        }
    }

    // ------------------------------
    // Save metadata
    // ------------------------------
    TNamed meta("config_summary",
        Form("nEvent=%d, eCM=%.1f, seed=%d, jetR=%.3f, jetPt=[%.1f,%.1f], jetEtaMax=%.2f, chargedOnly=%d, nRBins=%d, rMin=%.3e, normalizeByBinWidth=%d",
             cfg.nEvent, cfg.eCM, cfg.seed, cfg.jetR, cfg.jetPtMin, cfg.jetPtMax,
             cfg.jetEtaMax, (int)cfg.chargedOnly, cfg.nRBins, cfg.rMin, (int)cfg.normalizeByBinWidth));
    meta.Write();

    TParameter<long long> parNJets("nAcceptedJets", nAcceptedJets);
    parNJets.Write();

    // ------------------------------
    // Write output
    // ------------------------------
    fout->cd();
    hJetPt->Write();
    hJetEta->Write();
    hNConst->Write();
    hMeanEEC->Write();
    hVarEEC->Write();
    hRelFluc->Write();
    hCov->Write();
    hCorr->Write();
    tree->Write();

    fout->Close();

    pythia.stat();

    cout << "\nDone.\n";
    cout << "Accepted jets = " << nAcceptedJets << "\n";
    cout << "Output written to: " << cfg.outFileName << "\n\n";

    return 0;
}
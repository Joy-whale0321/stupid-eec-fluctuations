# plot_eec_fluct.py
#
# Read jet_eec_fluct.root and make quick plots.
#
# Run:
# python plot_eec_fluct.py jet_eec_fluct.root

import sys
import ROOT

ROOT.gROOT.SetBatch(True)

if len(sys.argv) < 2:
    print("Usage: python plot_eec_fluct.py jet_eec_fluct.root")
    sys.exit(1)

fname = sys.argv[1]
f = ROOT.TFile.Open(fname, "READ")
if not f or f.IsZombie():
    print(f"ERROR: cannot open {fname}")
    sys.exit(1)

hMean = f.Get("hMeanEEC")
hVar  = f.Get("hVarEEC")
hRel  = f.Get("hRelFluc")
hCov  = f.Get("hCov")
hCorr = f.Get("hCorr")

if not hMean:
    print("ERROR: hMeanEEC not found")
    sys.exit(1)

# -------- mean EEC --------
c1 = ROOT.TCanvas("c1", "mean EEC", 800, 700)
c1.SetLogx()
c1.SetLogy()
hMean.SetLineWidth(2)
hMean.Draw("hist")
c1.SaveAs("mean_eec.pdf")

# -------- variance --------
c2 = ROOT.TCanvas("c2", "variance", 800, 700)
c2.SetLogx()
c2.SetLogy()
hVar.SetLineWidth(2)
hVar.Draw("hist")
c2.SaveAs("var_eec.pdf")

# -------- relative fluctuation --------
c3 = ROOT.TCanvas("c3", "relative fluctuation", 800, 700)
c3.SetLogx()
hRel.SetLineWidth(2)
hRel.Draw("hist")
c3.SaveAs("relfluc_eec.pdf")

# -------- covariance --------
c4 = ROOT.TCanvas("c4", "covariance", 850, 750)
c4.SetLogx()
c4.SetLogy()
hCov.Draw("colz")
c4.SaveAs("cov_eec.pdf")

# -------- correlation --------
c5 = ROOT.TCanvas("c5", "correlation", 850, 750)
c5.SetLogx()
c5.SetLogy()
hCorr.SetMinimum(-1.0)
hCorr.SetMaximum(1.0)
hCorr.Draw("colz")
c5.SaveAs("corr_eec.pdf")

print("Saved:")
print("  mean_eec.pdf")
print("  var_eec.pdf")
print("  relfluc_eec.pdf")
print("  cov_eec.pdf")
print("  corr_eec.pdf")
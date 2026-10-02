"""Publish one identity-bound streaming model; diagnostics require explicit opt-in."""

def main():
    import sys
    if not any(arg == "--bundle" or arg.startswith("--bundle=") for arg in sys.argv[1:]):
        from scripts.runtime.speaker_start import main as publish_application
        from pathlib import Path
        return publish_application(Path(__file__).resolve().parent,["--mode","publish","--skip-train",*sys.argv[1:]])
    import argparse
    from pathlib import Path
    from vcm.packaging import publish
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--report',type=Path)
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pcv-runtime',action='store_true',help='Package a portable synthetic PCV deployment bundle for CPU-only ARM64')
    p.add_argument('--portable-parity',type=Path,help='Complete development waveform parity receipt for enabled PCV publication')
    p.add_argument('--sample-manifest',type=Path,help='Endpoint smoke sample JSONL with pinned shard references')
    p.add_argument('--source-receipt',type=Path,help='Matching acquisition-complete.json')
    p.add_argument('--source-card',type=Path,help='Matching pinned dataset source card')
    p.add_argument('--qualification-receipt',type=Path,help='Bound passing endpoint qualification receipt')
    a=p.parse_args()
    if a.pcv_runtime:
        from scripts.pcv_delivery import package
        print(package(a.bundle,a.output,diagnostic=a.diagnostic,portable_parity=a.portable_parity,
            sample_manifest=a.sample_manifest,source_receipt=a.source_receipt,source_card=a.source_card,
            qualification_receipt=a.qualification_receipt))
    else:
        print(publish(a.bundle,a.output,a.report,a.diagnostic))

if __name__=='__main__': main()

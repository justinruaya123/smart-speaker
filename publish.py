"""Publish one identity-bound streaming model; diagnostics require explicit opt-in."""

def main():
    import argparse
    from pathlib import Path
    from vcm.packaging import publish
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--report',type=Path)
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(publish(a.bundle,a.output,a.report,a.diagnostic))

if __name__=='__main__': main()

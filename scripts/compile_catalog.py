"""Compile fictional finite catalogs. Output has no calibrated acoustic policy."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vcm.catalog import compile_phrases, dictionary, pronounce, PHONES
from vcm.integrity import load,save,digest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True); p.add_argument('--dictionary',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    words=dictionary(a.dictionary)
    commands=[{'phrase':phrase,'phones':pronounce(phrase,words),'command':command} for phrase,command in sorted(compile_phrases(load(a.catalog)).items())]
    save(a.output,{'version':1,'phone_inventory':['<blank>']+PHONES,'commands':commands,
                   'source_sha256':digest(a.catalog),'dictionary_sha256':digest(a.dictionary),
                   'acoustic_support':'not evaluated; existing policy invalid after catalog replacement'})

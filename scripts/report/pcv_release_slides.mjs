/** Reproducible, editable two-slide ME2 report.
 * Usage: node scripts/report/pcv_release_slides.mjs --summary release-summary.json --output dist/pcv-release-v1/slides
 * Optional: --name me2-report-r2 (new name is required for an existing final PPTX).
 * Runtime overrides: RUNTIME_NODE_MODULES, RUNTIME_PYTHON, PRESENTATIONS_SKILL_DIR.
 * Nested input fields are documented by DEFAULT_SUMMARY below. Missing values remain pending.
 */
import fs from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import { pathToFileURL, fileURLToPath } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((a,v,i,all) => v.startsWith('--') ? [...a,[v.slice(2),all[i+1]]] : a, []));
const workspaceDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const runtimeRoot = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies');
const packages = process.env.RUNTIME_NODE_MODULES || path.join(runtimeRoot,'node/node_modules');
process.env.RUNTIME_NODE_MODULES = packages;
const SKILL_DIR = process.env.PRESENTATIONS_SKILL_DIR || path.join(os.homedir(),'.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations');
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON || path.join(runtimeRoot,'python/python.exe');
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(path.join(packages,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { finalizePresentation, resolvePresentationFont } = await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);

export const DEFAULT_SUMMARY = {
  title: 'ME2 – Voice Controlled Smart Device',
  subtitle: 'Always-on keyword + intent recognition at the edge · no cloud round-trip',
  author: '[Author / team]', date: '2 October 2026',
  status: 'Candidate evidence. Acceptance disabled: quality gates failed. Pi validation pending.',
  qualified: false, physical_pi_measured: false,
  architecture: ['Microphone · 16 kHz · 40-bin log-mel', 'Endpoint encoder · 6 blocks × 128 channels', 'Intent head', 'Finite slots', 'Validated JSON frontend', 'Raspberry Pi 4'],
  model: { name:'context128 INT8', parameters:'0.935 M', weights:'1.558 MB INT8', frontend:'10 ms input hop, 20 ms output grid'},
  dataset: {source:'HF: airimonda/ai231-me2-voice-commands',sourceShort:'HF voice commands (pinned)',revision:'e3e203b360d5c3205d59f99ad65cd380f9c0f517',hours:'Not audited',utterances:'56,556 rows',speakers:'Pending audited count',intents:'18',slots:'finite typed roles',access:'HF access terms',license:'Source-dependent licenses'},
  training: {compute:'Local RTX 4080 SUPER',objective:'Supervision + teacher targets',optimizer:'AdamW, cosine schedule',steps:'96/epoch',loss:'see receipt',seeds:'903'},
  metrics: {human:'473/609 (77.67%)',synthetic:'3,286/3,368 (97.57%)',humanAccepted:'0/609 (0.00%)',syntheticAccepted:'0/3,368 (0.00%)',baseline:'197/240, 71/93 phrases, 43 perfect',keyword:'Pending',intent:'Pending',falseAccept:'Pending',latencyP95:'Pending',rtf:'Pending',runtime:'ONNX Runtime',threads:'Pending'},
  release: {repository:'[Public repository URL pending]',codeLicense:'MIT requested; publication/license unverified',modelArtifact:'Candidate INT8 ONNX; release URL pending',modelLicense:'Model license review pending',reproduction:'Pending release',logs:'Existing receipts',piScript:'Pending Pi run',heldout:'Coverage gaps',baseline:'Partial comparison'},
  sources: ['docs/EXPERIMENT-pcv-final-qualification-v1.md','experiments/pcv-final-qualification-v1/results.json','experiments/pcv-final-qualification-v1/development-selection.json']
};
function merge(a,b) { const c={...a}; for(const [k,v] of Object.entries(b||{})) c[k]=v && typeof v==='object' && !Array.isArray(v) ? merge(a[k]||{},v) : v; return c; }
const data=merge(DEFAULT_SUMMARY,args.summary?JSON.parse(await fs.readFile(path.resolve(args.summary),'utf8')):{});
// Keep full source fields in notes. Slide copy only removes redundant labels or
// uses explicit concise statuses; it never converts raw accuracy into acceptance.
const rawMetric=value=>String(value).replace(/^Raw candidate\s+/i,'');
const computeShort=String(data.training.compute).replace(/\s*\(local; no A100\)/i,'').replace(/^NVIDIA\s+(GeForce\s+)?/i,'');
const sourceShort=String(data.dataset.sourceShort).replace(/^https:\/\/huggingface.co\/datasets\//,'').replace('airimonda/','airimonda/\n');
const sourceLocation=String(data.dataset.source).replace(/^https:\/\/huggingface.co\/datasets\//,'HF: ');
const objectiveShort=/matched hard\s*\/\s*teacher arms/i.test(data.training.objective)?'CTC + intent/target\nHard vs teacher arms':data.training.objective;
const stepsShort=/^\d+ student updates/.test(data.training.steps)?`${Number.parseInt(data.training.steps,10).toLocaleString('en-US')} updates / histories`:`${data.training.steps} / ${data.training.loss}`;
const qualificationText=data.qualified===true?'Qualified for Pi validation':data.qualified===false?'Diagnostic candidate: acceptance gates failed':'Qualification pending';
const piText=data.physical_pi_measured===true?'Pi measurements recorded':'Pi validation pending';
const reviewer={
 reproduction:/not verified|unverified|pending/i.test(data.release.repository)?'Repo unverified':/release\.py/i.test(data.release.reproduction)?'Recipe provided':'See notes',
 dataset:'Source terms',
 logs:/histories|receipts|logs/i.test(data.release.logs)?'Evidence retained':'See notes',
 piScript:data.physical_pi_measured===true?'Measured; see notes':'Pi pending',
 heldout:/published.*retained/i.test(data.release.heldout)?'Published splits':/coverage gaps/i.test(data.release.heldout)?'Coverage gaps':'See notes',
 baseline:/matched/i.test(data.release.baseline)?'Matched arms':/partial/i.test(data.release.baseline)?'Partial comparison':'See notes',
 ...data.release.statuses,
};
const output=path.resolve(args.output||path.join(workspaceDir,'dist/pcv-release-v1/slides'));
const name=args.name||'me2-report';
if(!/^[a-zA-Z0-9_-]+$/.test(name)) throw new Error('--name must contain letters, digits, underscores or hyphens');
const scratchRoot=path.join(workspaceDir,'.cache/pcv-release-v1/slides');
await fs.mkdir(scratchRoot,{recursive:true}); await fs.mkdir(output,{recursive:true});
// Final filenames may repeat in distinct release directories. Validation receipts
// are exclusive files, so every invocation needs its own private scratch folder.
const tmp=await fs.mkdtemp(path.join(scratchRoot,`${name}-`));
const finalPdf=path.join(output,`${name}.pdf`);
try { await fs.access(finalPdf); throw new Error(`Preserve existing PDF: ${finalPdf}`); }
catch(error) { if(error.code!=='ENOENT') throw error; }
const family=resolvePresentationFont({fontFamily:'Arial'});
const C={black:'#121820',blue:'#2854A0',green:'#80B342',header:'#1C2C43',line:'#D6DBE2',light:'#F5F6FA'};
const presentation=Presentation.create({slideSize:{width:1200,height:820}});
function txt(slide,text,x,y,w,h,size=23,bold=false,color=C.black,align='left') {
 const sh=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=String(text); sh.text.style={typeface:family,fontSize:size,bold,color,alignment:align,autoFit:'none'}; return sh;
}
function line(slide,x,y,w,color=C.green) {slide.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:3},fill:color,line:{fill:'none',width:0}});}
function table(slide,values,x,y,width,rowHeight,colWidths,fontSize=22,header=true) {
 const t=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height:rowHeight*values.length,columnWidths:colWidths,values});
 t.borders.assign({style:'solid',fill:C.line,width:0.6});
 for(let r=0;r<values.length;r++) {t.rows[r].height=rowHeight; for(let c=0;c<values[r].length;c++) {
  const cell=t.getCell(r,c); cell.fill=header&&r===0?C.header:(r%2?C.light:'#FFFFFF');
  cell.text.style={typeface:family,fontSize,bold:header&&r===0,color:header&&r===0?'#FFFFFF':C.black,autoFit:'none'};
 }}
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:7,right:7,top:3,bottom:3},anchor:'center'});
 return t;
}
function block(slide,label,x,y,w,h,fill,border) {
 const s=slide.shapes.add({geometry:'roundRect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:border,width:1.2},borderRadius:4});
 s.text=label; s.text.style={typeface:family,fontSize:22,bold:true,color:C.black,alignment:'center',verticalAlignment:'middle',autoFit:'none'}; return s;
}
const a=presentation.slides.add(); a.background.fill='#FFFFFF';
txt(a,data.title,44,24,1112,44,36,true);
txt(a,data.subtitle,44,72,1112,32,24,true);
txt(a,`${data.date} · ${data.author} · Local training · ${piText}`,44,110,1112,29,20);
line(a,44,146,1112);
txt(a,'Model',44,160,532,32,27,true,C.blue);
block(a,data.architecture[0],44,198,532,34,'#EDF2F8','#9EAFC5');
block(a,data.architecture[1],44,238,532,43,'#EAF5E9','#84A875');
block(a,data.architecture[2],44,287,263,34,'#FFF4DB','#C8A65D');
block(a,String(data.architecture[3]).replace(/^Finite command scorer$/,'Finite scorer'),313,287,263,34,'#FFF4DB','#C8A65D');
block(a,data.architecture[4],44,327,343,34,'#F0EAF8','#9C87B7');
block(a,data.architecture[5],393,327,183,34,'#EEF4E8','#88AF60');
txt(a,'Complete utterance after VAD. No streaming cache.',44,368,532,26,18);
table(a,[['Item','Value'],['Parameters / weights',`${data.model.parameters} / ${data.model.weights}`]],44,403,532,37,[225,307],21);
txt(a,'Validation on the Raspberry Pi 4',44,493,532,32,27,true,C.blue);
table(a,[['Item','Value'],['Keyword / intent acc.',`${data.metrics.keyword} / ${data.metrics.intent}`],['False-accept rate',data.metrics.falseAccept],['Latency p95 / RTF',`${data.metrics.latencyP95} / ${data.metrics.rtf}`],['Runtime / threads',`${data.metrics.runtime} / ${data.metrics.threads}`]],44,534,532,34,[252,280],21);
txt(a,'Pi evidence: metrics.json + REPORT.md',44,711,532,26,18,false,'#555F6B');
txt(a,'Dataset',605,160,551,32,27,true,C.blue);
table(a,[['Item','Value'],['Source',sourceShort],['Hours / utts',`${data.dataset.hours} / ${data.dataset.utterances}`],['Speakers',data.dataset.speakers],['Labels',`${data.dataset.intents} intents / ${data.dataset.slots}`]],605,198,551,39,[180,371],20);
txt(a,'Local training',605,421,551,32,27,true,C.blue);
table(a,[['Item','Value'],['Compute',computeShort],['Objective',objectiveShort],['Optimizer',data.training.optimizer],['Steps / loss',stepsShort]],605,460,551,31,[160,391],19);
table(a,[['Test exact','Human','Synthetic'],['Raw',rawMetric(data.metrics.human),rawMetric(data.metrics.synthetic)],['Accepted',data.metrics.humanAccepted,data.metrics.syntheticAccepted]],605,649,551,31,[151,200,200],18);
line(a,44,748,1112);
txt(a,`${qualificationText}. ${piText}.`,44,759,1112,43,20,true,data.qualified===true?'#355F1B':'#735A16');
a.speakerNotes.textFrame.setText(`Candidate identity: ${data.model.name}. ${data.model.frontend}. Qualification: ${data.qualified}. Summary status: ${data.status}. Raw exact-command results precede acceptance filtering. Accepted exact-command counts use the positive-command denominator and report recall, not accepted precision. Neither row is a keyword accuracy measure. Physical Pi measured: ${data.physical_pi_measured}. Training compute: ${data.training.compute}. Sources:\n${data.sources.join('\n')}\nDataset: https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/tree/${data.dataset.revision}\nComplete input summary (including unabridged training, reviewer and publication details):\n${JSON.stringify(data,null,2)}`);
const b=presentation.slides.add(); b.background.fill='#FFFFFF';
txt(b,'To Be Submitted',44,30,1112,52,40,true); line(b,44,98,1112);
const repoShort=String(data.release.repository).replace(/\s*\(public status not verified\)/i,'');
const datasetTerms=/no DOI verified/i.test(data.dataset.access)?'Source-dependent terms; no DOI verified':'Source terms and access: see notes';
const submissions=table(b,[['GitHub repository',`${repoShort}\n${data.release.codeLicense}`],['Dataset location',`${sourceLocation}\nRev ${data.dataset.revision.slice(0,12)} · ${datasetTerms}`],['Local training',`${computeShort} (local)\nSeeds ${data.training.seeds}`],['Model weights',`${data.release.modelArtifact}\n${data.release.modelLicense}`]],44,119,1112,70,[270,842],22,false);
for(let r=0;r<4;r++){submissions.getCell(r,0).fill='#FAF0D8';submissions.getCell(r,0).text.style={typeface:family,fontSize:24,bold:true,color:'#634C19'};}
txt(b,'What the reviewer will look for',44,423,1112,38,28,true,C.blue);
table(b,[['#','Item','Status'],['1','Public repo, one-command reproduction',reviewer.reproduction],['2','Dataset license, citation and fixed revision',reviewer.dataset],['3','Training logs and selected checkpoint',reviewer.logs],['4','Pi 4 latency reproduced with the posted script',reviewer.piScript],['5','Held-out tests and unseen speakers',reviewer.heldout],['6','Baseline of comparable size',reviewer.baseline]],44,471,1112,39,[48,792,272],23);
txt(b,`${piText}. Full publication, license and reproduction details are in the notes.`,44,763,1112,30,19,false,'#555F6B');
b.speakerNotes.textFrame.setText(`Qualification: ${data.qualified}. ${data.status}. Physical Pi measured: ${data.physical_pi_measured}. Dataset access and redistribution depend on upstream source terms. Training compute: ${data.training.compute}. Sources:\n${data.sources.join('\n')}\nHuman phrase baseline: ${data.metrics.baseline}. Raw phrase correctness differs from calibrated acceptance. Full dataset revision: ${data.dataset.revision}.\nFull release details:\n${JSON.stringify(data.release,null,2)}\nFull training details:\n${JSON.stringify(data.training,null,2)}\nFull dataset details:\n${JSON.stringify(data.dataset,null,2)}`);
const candidatePath=path.join(tmp,'candidate.pptx'); const finalPath=path.join(output,`${name}.pptx`);
await fs.writeFile(path.join(tmp,'summary.json'),JSON.stringify(data,null,2));
await(await PresentationFile.exportPptx(presentation)).save(candidatePath);
await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:RUNTIME_PYTHON,integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),explicitTotalSlideCount:2,requiredNativeTableOwnerSlides:[1,2],fontPolicy:{basis:'design',families:[family]},layoutArgs:['--expected-slide-size-emu','11430000,7810500','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','1','--require-native-table-slide','2'],verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation.json')});
const imported=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<2;i++){const slide=imported.slides.getItem(i);const png=await imported.export({slide,format:'png',scale:2});await fs.writeFile(path.join(tmp,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));const layout=await slide.export({format:'layout'});await fs.writeFile(path.join(tmp,`slide-${i+1}.layout.json`),await layout.text());}
// Raster PDF companion uses the inspected PPTX render. Editable tables, diagram,
// complete source notes and text remain in the PPTX, not in the PDF images.
const pdfCandidate=path.join(tmp,'candidate.pdf');
const pdfCode=String.raw`
import hashlib, io, json, sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader
folder, target = Path(sys.argv[1]), Path(sys.argv[2])
page_size = (900, 615)
buffer = io.BytesIO()
pdf = canvas.Canvas(buffer, pagesize=page_size, pageCompression=1)
pdf.setTitle('ME2 - Voice Controlled Smart Device')
pdf.setSubject('Visual companion to the editable two-slide PowerPoint; slide images, not editable PDF text')
for number in (1, 2):
    pdf.drawImage(ImageReader(str(folder / f'slide-{number}.png')), 0, 0, width=page_size[0], height=page_size[1])
    pdf.showPage()
pdf.save()
payload=buffer.getvalue()
reader=PdfReader(io.BytesIO(payload))
if len(reader.pages)!=2: raise ValueError('PDF must contain exactly two pages')
for page in reader.pages:
    if tuple(float(v) for v in (page.mediabox.width,page.mediabox.height))!=page_size:
        raise ValueError('PDF page size changed')
target.write_bytes(payload)
print(json.dumps({'pages':len(reader.pages),'page_size_pt':page_size,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'visual_companion':True,'editable':False}))
`;
const pdfResult=spawnSync(RUNTIME_PYTHON,['-c',pdfCode,tmp,pdfCandidate],{encoding:'utf8',windowsHide:true});
if(pdfResult.status!==0) throw new Error(`PDF export failed: ${pdfResult.stderr||pdfResult.error}`);
const pdfMetadata=JSON.parse(pdfResult.stdout.trim());
await fs.copyFile(pdfCandidate,finalPdf,fsConstants.COPYFILE_EXCL);
const pdftoppm=process.env.PDFTOPPM||path.join(runtimeRoot,'native/poppler/Library/bin/pdftoppm.exe');
const pdfRender=spawnSync(pdftoppm,['-r','120','-png',finalPdf,path.join(tmp,'pdf-slide')],{encoding:'utf8',windowsHide:true});
if(pdfRender.status!==0) throw new Error(`PDF render failed: ${pdfRender.stderr||pdfRender.error}`);
await fs.writeFile(path.join(tmp,'pdf.validation.json'),JSON.stringify({...pdfMetadata,path:finalPdf,renderedPages:['pdf-slide-1.png','pdf-slide-2.png']},null,2));
console.log(JSON.stringify({pptx:finalPath,pdf:finalPdf,pdfMetadata,previewDirectory:tmp,slides:2}));

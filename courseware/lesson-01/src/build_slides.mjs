// Authoring requires the Codex bundled @oai/artifact-tool runtime. Learners use the PPTX directly.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const root=process.env.LESSON_ROOT;
const skill=process.env.PRESENTATION_SKILL;
const build=process.env.DECK_BUILD;
if(![root,skill,build].every(x=>x&&path.isAbsolute(x)))throw new Error('Set LESSON_ROOT, PRESENTATION_SKILL, DECK_BUILD to absolute paths');
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font='Heiti TC';
const content=JSON.parse(await fs.readFile(path.join(root,'slides/content.json'),'utf8'));
const results=JSON.parse(await fs.readFile(path.join(root,'reference/baseline-v1/results.json'),'utf8'));
const pres=Presentation.create({slideSize:{width:1280,height:720}});
function text(slide,value,left,top,width,height,size=32,color='#172F46',bold=false){
 const shape=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
 shape.text=value;shape.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none'};return shape;
}
for(let i=0;i<content.length;i++){
 const d=content[i];const slide=pres.slides.add();slide.background.fill=i===0?'#132D44':'#FFFFFF';
 const color=i===0?'#FFFFFF':'#172F46';
 text(slide,d.title,72,50,1136,i===0?178:85,i===0?60:44,color,true);
 if(d.chart){
  const chart=slide.charts.add('bar',{position:{left:110,top:170,width:1060,height:410},
   categories:['已知 macro-F1','已知覆盖率','未知误收率'],
   series:[{name:'规则',values:['known_macro_f1','known_coverage','unknown_false_accept_rate'].map(k=>Number(results.metrics.rules[k].toFixed(6))),fill:'#315C9B'},
           {name:'TF-IDF＋拒识',values:['known_macro_f1','known_coverage','unknown_false_accept_rate'].map(k=>Number(results.metrics.tfidf[k].toFixed(6))),fill:'#C47B27'}],
   barOptions:{direction:'column',grouping:'clustered'},hasLegend:true,
   legend:{position:'bottom',textStyle:{fontSize:22,typeface:font}},
   dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0%',textStyle:{fontSize:22,typeface:font}},
   xAxis:{textStyle:{typeface:font,fontSize:22}},yAxis:{minimumScale:0,maximumScale:1,numberFormatCode:'0%',textStyle:{typeface:font,fontSize:20}}});
  applyPresentationChartFont(chart,{fontFamily:font});
  text(slide,'合成测试：已知240条、未知40条；拒识的已知请求仍计错',72,614,1136,44,24);
 }else{
  const start=i===0?290:196;const gap=i===0?90:100;
  for(let j=0;j<d.lines.length;j++)text(slide,d.lines[j],76,start+j*gap,1128,80,i===0?32:30,color);
 }
 text(slide,String(i+1).padStart(2,'0'),1165,667,55,28,18,i===0?'#ADC2D1':'#697A86');
 slide.speakerNotes.textFrame.setText(d.notes+'\n第01课教学包；讲稿详见教师讲义.md。');
}
await fs.mkdir(build,{recursive:true});
const candidate=path.join(build,'lesson01-candidate.pptx');
await(await PresentationFile.exportPptx(pres)).save(candidate);
const finalPath=path.join(root,'slides/第01课-业务问题建模与传统NLP基线.pptx');
const validation=await finalizePresentation({workspaceDir:path.dirname(root),candidatePath:candidate,finalPath,
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 requiredNativeChartOwnerSlides:[14],requiredNativeTableOwnerSlides:[],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation.json')});
// Render finalized package, not only the in-memory draft.
const {FileBlob}=await import('@oai/artifact-tool');
const finalPres=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<content.length;i++){
 const s=finalPres.slides.items[i];
 const blob=await finalPres.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log(JSON.stringify({finalPath,slides:content.length}));

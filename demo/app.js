(function(){
const $=id=>document.getElementById(id);
const NS='http://www.w3.org/2000/svg';
const tip=$('tip');
function showTip(text,x,y){tip.textContent=text;tip.hidden=false;const r=tip.getBoundingClientRect();
  tip.style.left=Math.min(Math.max(8,x+12),innerWidth-r.width-8)+'px';tip.style.top=Math.max(8,y-r.height-10)+'px'}
function hideTip(){tip.hidden=true}
function el(tag,attrs,parent,text){const n=document.createElementNS(NS,tag);for(const k in attrs)n.setAttribute(k,attrs[k]);if(text!=null)n.textContent=text;if(parent)parent.appendChild(n);return n}
function hit(node,label){node.setAttribute('tabindex','0');node.setAttribute('role','img');node.setAttribute('aria-label',label);node.classList.add('hit');
  node.addEventListener('pointermove',e=>showTip(label,e.clientX,e.clientY));node.addEventListener('pointerleave',hideTip);
  node.addEventListener('focus',()=>{const r=node.getBoundingClientRect();showTip(label,r.left+r.width/2,r.top)});node.addEventListener('blur',hideTip)}
function svg(host,w,h,label){const s=el('svg',{viewBox:`0 0 ${w} ${h}`,role:'group','aria-label':label});host.appendChild(s);return s}
function legend(host,items){const d=document.createElement('p');d.className='legend';items.forEach(([c,t])=>{const s=document.createElement('span');const i=document.createElement('i');i.style.background=`var(${c})`;s.append(i,t);d.append(s)});host.appendChild(d)}

/* gallery */
const wipe=$('wipe'),range=$('wipe-range');
range.addEventListener('input',()=>wipe.style.setProperty('--pos',range.value+'%'));
$('overlay-toggle').addEventListener('change',e=>{$('img-overlay').hidden=!e.target.checked});
let gallery=[];
function select(i){const s=gallery[i];
  [['cond',`Canny edge map for: ${s.prompt}`],['on',`Generated with control on, following the edge map, for: ${s.prompt}`],['off',`Generated with control off, same seed, ignoring the edge map, for: ${s.prompt}`],['orig',`Original COCO photo the edges came from: ${s.prompt}`]].forEach(([k,a])=>{const im=$('img-'+k);im.src=`img/s${i}-${k}.webp`;im.alt=a});
  $('img-overlay').src=`img/s${i}-cond.webp`;
  $('prompt').textContent=s.prompt;$('r-off').textContent=s.recallOff.toFixed(3);$('r-on').textContent=s.recallOn.toFixed(3);$('orig-id').textContent='COCO val2017 #'+s.cocoId;
  document.querySelectorAll('#strip button').forEach((b,j)=>b.setAttribute('aria-pressed',j===i))}
fetch('data/gallery.json').then(r=>r.json()).then(g=>{gallery=g;const strip=$('strip');
  g.forEach((s,i)=>{const b=document.createElement('button');b.type='button';b.setAttribute('aria-pressed',i===0);b.setAttribute('aria-label',`Sample ${i+1}: ${s.prompt}`);
    const im=new Image();im.src=`img/s${i}-on.webp`;im.alt='';im.width=253;im.height=253;b.append(im);b.addEventListener('click',()=>select(i));strip.append(b);
    const li=document.createElement('li');li.append(`#${s.cocoId}: `);const a1=document.createElement('a');a1.href=s.source;a1.textContent='Flickr source';const a2=document.createElement('a');a2.href=s.licenseUrl;a2.textContent=s.license;li.append(a1,', ',a2);$('credits').append(li)});
  recallChart(g)});

/* metrics */
function metricsChart(on,off){const host=$('chart-metrics');legend(host,[['--off','Control off (scale 0)'],['--acc','Control on (scale 1)']]);
  const rows=[['Edge recall','canny_edge_recall'],['Edge F1','canny_edge_f1'],['SSIM','canny_ssim']];
  const W=560,H=250,L=88,R=44,T=4,rowH=(H-T-22)/3,x=v=>L+(W-L-R)*v;
  const s=svg(host,W,H,'Bar chart of edge recall, edge F1 and SSIM with control on and off');
  const g=el('g',{class:'grid'},s),a=el('g',{class:'axis'},s);
  [0,.25,.5,.75,1].forEach(v=>{el('line',{x1:x(v),x2:x(v),y1:T,y2:H-22},g);el('text',{x:x(v),y:H-6,'text-anchor':'middle'},a,v)});
  rows.forEach(([name,key],i)=>{const y=T+i*rowH+8,bh=(rowH-22)/2;
    el('text',{x:0,y:y+bh+1,class:'lab2','dominant-baseline':'middle'},s,name);
    [[off[key],'--off','off'],[on[key],'--acc','on']].forEach(([v,c,lab],j)=>{const by=y+j*(bh+2);
      el('rect',{x:L,y:by,width:x(v)-L,height:bh,rx:2,fill:`var(${c})`},s);
      el('text',{x:x(v)+6,y:by+bh/2,class:'lab','dominant-baseline':'middle'},s,v.toFixed(2));
      hit(el('rect',{x:L,y:by-1,width:W-L,height:bh+2},s),`${name}, control ${lab}: ${v.toFixed(3)}`)})})}
Promise.all(['data/eval_report.json','data/eval_baseline.json'].map(u=>fetch(u).then(r=>r.json()))).then(([on,off])=>metricsChart(on,off));

/* per-sample dumbbell */
function recallChart(g){const host=$('chart-recall');legend(host,[['--off','Control off'],['--acc','Control on']]);
  const order=g.map((s,i)=>i).sort((a,b)=>g[b].recallOn-g[a].recallOn);
  const narrow=host.clientWidth<520;
  const W=narrow?420:960,rowH=narrow?38:26,T=4,H=T+rowH*8+24,L=narrow?0:300,R=20,x=v=>L+(W-L-R)*v;
  const s=svg(host,W,H,'Dumbbell chart of edge recall per sample, control off and on; recall rises on all eight');
  const gr=el('g',{class:'grid'},s),a=el('g',{class:'axis'},s);
  [0,.25,.5,.75,1].forEach(v=>{el('line',{x1:x(v),x2:x(v),y1:T,y2:H-22},gr);el('text',{x:x(v),y:H-6,'text-anchor':v===0?'start':v===1?'end':'middle'},a,v)});
  order.forEach((i,row)=>{const d=g[i],y=T+row*rowH+rowH/2+(narrow?7:0),max=narrow?52:38,short=d.prompt.length>max?d.prompt.slice(0,max-1)+'…':d.prompt;
    el('text',{x:0,y:narrow?y-13:y,class:'lab2','dominant-baseline':'middle'},s,short);
    el('line',{x1:x(d.recallOff),x2:x(d.recallOn),y1:y,y2:y,stroke:'var(--line)','stroke-width':3},s);
    el('circle',{cx:x(d.recallOff),cy:y,r:5,fill:'var(--off)',stroke:'var(--bg)','stroke-width':2},s);
    el('circle',{cx:x(d.recallOn),cy:y,r:5,fill:'var(--acc)',stroke:'var(--bg)','stroke-width':2},s);
    hit(el('rect',{x:0,y:T+row*rowH,width:W,height:rowH},s),`${d.prompt} Recall ${d.recallOff.toFixed(3)} off, ${d.recallOn.toFixed(3)} on`)})}

/* zero conv, recorded in scripts/plot_training.py */
(function(){const host=$('chart-zero');const steps=[2000,4000,12000];
  const series=[['Down zero-convs','--s1',[3.98e-4,5.58e-4,1.07e-3]],['Mid zero-conv','--s2',[3.47e-4,4.93e-4,8.87e-4]],['Cond-embed out','--s3',[1.09e-4,1.40e-4,1.77e-3]]];
  legend(host,series.map(([n,c])=>[c,n]));
  const W=560,H=280,L=56,R=16,T=10,B=26,x=v=>L+(W-L-R)*(v/12000),lo=-4,hi=-1,y=v=>T+(H-T-B)*(1-(Math.log10(v)-lo)/(hi-lo));
  const s=svg(host,W,H,'Line chart on a log scale of mean absolute zero-conv weight at 2k, 4k and 12k steps');
  const g=el('g',{class:'grid'},s),a=el('g',{class:'axis'},s);
  [1e-4,1e-3,1e-2,1e-1].forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v)},g);el('text',{x:L-6,y:y(v)+4,'text-anchor':'end'},a,v.toExponential(0))});
  steps.forEach(v=>el('text',{x:x(v),y:H-6,'text-anchor':v===12000?'end':'middle'},a,(v/1000)+'k'));
  el('line',{x1:L,x2:W-R,y1:y(3.5e-2),y2:y(3.5e-2),stroke:'var(--ink2)','stroke-dasharray':'4 4'},s);
  el('text',{x:W-R,y:y(3.5e-2)-6,'text-anchor':'end',class:'lab2'},s,'Normal conv, 3.5e-2');
  series.forEach(([n,c,vals])=>{el('polyline',{points:vals.map((v,i)=>x(steps[i])+','+y(v)).join(' '),fill:'none',stroke:`var(${c})`,'stroke-width':2},s);
    vals.forEach((v,i)=>{el('circle',{cx:x(steps[i]),cy:y(v),r:4,fill:`var(${c})`,stroke:'var(--bg)','stroke-width':2},s);
      hit(el('circle',{cx:x(steps[i]),cy:y(v),r:11},s),`${n} at ${steps[i]/1000}k steps: ${v.toExponential(2)}`)})})})();
})();

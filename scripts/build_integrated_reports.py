from pathlib import Path
import re,shutil,json
ROOT=Path('/home/hyein/ai-benchmark');PUB=Path('/home/hyein/ai-benchmark-publication');RAM=121.68751525878906
MODELS=[
 ('esm2','ESM-2','단백질 아미노산 서열 → 단백질당1280차원 임베딩 또는 fluorescence 예측. 체크포인트 esm2_t33_650M_UR50D(명목650M).',['ESM2_20261027_164300_125163.md','ESM2_batch_memory.md'],[
 ('추론',64.41565036773682,'실제900~950아미노산 단백질384개. GPU토큰 배열384×952(BOS/EOS포함),FP32.','80% 미확인; 다음배치를예상예약량guard로건너뜀'),
 ('부분 파인튜닝',3.6230783462524414,'실제237아미노산 서열128개 학습. 토큰예산4096 가변배치,FP32; 마지막Transformer block·최종LayerNorm·회귀head만학습.','학습배치확대실험 미실시; 전체모델학습값 아님')]),
 ('colabfold','AlphaFold2 — ColabFold','단백질 서열과 MSA → 3D 원자좌표·신뢰도. 단량체 PTM과 Multimer를기본보고서에서구분한다.',['AlphaFold2_ColabFold.md','Batch_memory_followup.md'],[
 ('기본 추론의 누적 관측값',2.705740213394165,'단백질247아미노산 조건까지진행. 단백질1개씩,5개weight set 순차예측,3recycle,BF16계산/FP32가중치. MSA축은부가정보이며단백질batch가아님.','JAX프로세스누적peak(이전query/컴파일포함); CUDA실행구간peak와직접비교불가'),
 ('별도 배치 추론',2.3083367347717285,'각115아미노산인실제단백질4개,model_1 고정,기존MSA·3recycle·FP32,JAX vmap.','80% 미확인; 실제4query풀종료. CLI기본batch기능이아닌별도구현'),
 ('학습·파인튜닝',None,'이번환경에서수행하지않음.','80% 조건 미조사; 미실시를학습불가능으로단정하지않음')]),
 ('scvi','scVI','raw count의 세포×유전자 행렬과 실제생물학적batch → 세포당10차원 latent. 처음부터학습한SCVI 모델.',['scVI.md','Batch_memory_followup.md'],[
 ('추론',.26955747604370117,'한번에10176세포×2000유전자,FP32. 기존저장모델의latent posterior mean.','80% 미확인; 확보한학습세포10176개전체batch까지시험'),
 ('처음부터 학습',.09743928909301758,'학습세포10176개를128세포×2000유전자씩분할,400epoch,FP32.','파인튜닝아님; 학습batch확대 미실시')]),
 ('cellpose','Cellpose','이미지 → 핵/세포별 instance mask. cpsam_v2(약304.6M),영상개수와동시GPU타일개수를구분한다.',['Cellpose.md','Cellpose_batch_memory.md','Cellpose_brain_NucMM.md'],[
 ('이미지 타일 추론',43.39664602279663,'실제520×696·1채널이미지32개→256×256타일384개동시처리. GPU입력384×1×256×256,BF16.','80% 미확인; 다음배치576은예상예약량guard로미실행'),
 ('정답 마스크 기반 파인튜닝',7.9846320152282715,'실제train16/val8/test16이미지. 학습GPU입력1×3×256×256(FP32),100epoch.','학습batch1 관측값; 학습batch확대 미실시'),
 ('뇌 조직 3D 추가 추론',1.4913725852966309,'NucMM-M micro-CT 192×192×192복셀4patch. tilebatch8·256타일,BF16,최종3D마스크후처리포함.','일반뇌MRI아님. AP50평균0.422; 기존타일network시간과범위다름')]),
 ('dnabert2','DNABERT-2','DNA 염기서열 → BPE토큰 → 임베딩/18개histone 표지 예측. 염기수와토큰수는다르다.',['DNABERT2.md','Batch_memory_followup.md'],[
 ('추론',60.0069,'실제4000bp 서열512개,FP32. input_ids/attention_mask 각각512×851(pool최대길이로padding).','80% 미확인; 다음배치예상예약량guard로중단'),
 ('기존 지도 학습',10.571216583251953,'BEND실제1000서열·18표지라벨,4000bp·batch8·FP32·1epoch. 실제697~851토큰.','학습batch확대 미실시. 새128개16step자원probe와구분')]),
 ('ntv2','Nucleotide Transformer V2','DNA서열 → 주로6-mer 토큰 → 임베딩 또는enhancer 분류. 공개500M checkpoint.',['NTv2.md','Batch_memory_followup.md'],[
 ('추론',21.592,'실제6000nt 서열128개. GPU입력128×1001토큰,FP32.','80% 미확인; 확보한고유서열128개한계'),
 ('전체 backbone 파인튜닝',9.252224445343018,'199~200nt enhancer서열,train512/val128/test400. batch8·FP32·2epoch,분류head와backbone학습.','학습batch확대 미실시')]),
 ('clair3','Clair3','long-read BAM+참조유전체 → 변이후보특징배열 → 변이확률/VCF. batch는read수가아닌후보feature 배열수.',['Clair3.md','Batch_memory_followup.md'],[
 ('pileup 후보 추론',4.1114301681518555,'실제HG003 ONT 후보feature5708개. 각후보33×18,입력5708×33×18. FP32모델,원본feature dtype은각audit에기록.','80% 미확인; 확보한고유feature pool끝'),
 ('full-alignment 후보 추론',1.499110221862793,'실제후보feature894개. 각후보89×33×8,입력894×89×33×8,FP32모델.','80% 미확인; 후보좌표를export하지못해고유위치수로주장하지않음'),
 ('학습·파인튜닝',None,'이번작업에서수행하지않음.','공식Train기능과이번실시범위를구분')]),
 ('rnafm','RNA-FM','RNA 염기서열 → 염기토큰 → 640차원 임베딩. 분류실험은본체를동결하고선형head만학습.',['RNAFM.md','Batch_memory_followup.md'],[
 ('기존 실제길이 혼합 추론',8.870131969451904,'train896개 RNA를한번에최대50개씩임베딩추출. 입력은배치별길이로padding,관측shape중최대50×1022토큰(최대1020nt). FP32.','이전체추출구간의peak; 어느개별batch에서peak인지별도분리하지않음. 80% 미확인'),
 ('고정512nt 배치 후속 추론',3.844,'512nt 실제RNA76개,입력76×514토큰(BOS/EOS포함),FP32.','고유76서열풀끝. 이값이전체실험최대라는뜻아님'),
 ('동결 임베딩의 분류head 학습',.43360376358032227,'추출된640차원벡터를batch8×640으로입력,28종RNA label,100epoch. 본체는학습하지않고GPU에남아있었음.','본체파인튜닝 미실시. 메모리에상주한동결본체도포함')]),
 ('openspliceai','OpenSpliceAI','DNA구간을4채널one-hot로변환 → 위치별splice site확률. 같은5개MANE10000 모델ensemble.',['OpenSpliceAI.md','Batch_memory_followup.md'],[
 ('5모델 ensemble 추론',1.463,'실제15000nt 구간156개. GPU입력156×4×15000,출력156×3×5000,FP32. 5모델순차계산/평균.','80% 미확인; 확보한고유구간156개한계'),
 ('단일 seed10 전체 파인튜닝',2.1602683067321777,'실제train96/val24/test36구간. 12×4×15000 입력,정답12×5000 위치라벨,FP32·10epoch.','5모델을함께학습한값아님; 학습batch확대 미실시')]),
 ('scgpt','scGPT','세포×유전자발현 행렬 → 세포별유전자ID·발현bin값 → 임베딩/세포유형. batch는세포행수. 최대1200토큰은CLS+유전자+padding자리.',['scGPT.md','scGPT_memory_batch.md','scGPT_realdata_full.md','scGPT_actual_300k.md'],[
 ('실제30만세포 추론',5.4772539138793945,'고유30만세포를256개씩처리. 세포당최대1200토큰,배치내실제최대길이padding,FP32.','80% 미확인; 30만세포를동시에GPU에올린값아님'),
 ('학습 용량 probe',100.75033235549927,'실제1000학습세포cache 중292세포×1200고정padding,FP32·AdamW·3step. 실제평균nonpad약306자리.','80% 도달 확인(82.79%). padding포함용량probe이며전체데이터파인튜닝과구분'),
 ('실제 전체 pancreas 파인튜닝',3.374143600463867,'train8480/val2120/test4218세포,학습batch8·FP32·최대1200·원래dynamic padding·1epoch.','큰batch284/256/224는메모리guard중단. 이조건에서80%성공을입증하지못함')])]
# Replace rounded placeholders with exact artifact values.
for m in MODELS:
 name=m[0]
 if name in ['dnabert2','ntv2','rnafm','openspliceai']:
  r=ROOT/name/'results';s=json.loads((r/(r/'BATCH_LATEST').read_text().strip()/'summary.json').read_text());exact=s['conditions'][-1]['memory']['peak_allocated_gib']
  index=1 if name=='rnafm' else 0;row=list(m[4][index]);row[1]=exact;m[4][index]=tuple(row)
def shifted(text):
 lines=text.splitlines();out=[];fence=False
 for line in lines:
  if line.startswith('> 최신 통합판:'):continue
  if line.lstrip().startswith(('```','~~~')):fence=not fence
  if not fence and re.match(r'^#{1,5} ',line):line='#'+line
  out.append(line)
 return '\n'.join(out)
for target in [ROOT,PUB]:
 entries=[]
 for model,title,meaning,parts,measurements in MODELS:
  R=target/model/'reports';out=[f'# {title} 최종 통합 보고서','','기본 실험과 모든 후속 실험을 이 문서에 함께 수록했다. 다른 기본/후속 문서를 따로 읽을 필요 없이 아래 요약과 본문을 보면 된다.','','## 1. 입력과 메모리 결론','',meaning,'','**목표는 메모리 할당80% 조건 확인**이다. 전체공유RAM121.688GiB의80%는약97.35GiB다. GPU-Util은별도연산활동지표이며이목표의성공판정에쓰지않는다.','', '|작업|관측메모리 peak GiB|공유RAM대비|해당 입력·배치·정밀도|80% 확인/제약|','|---|---|---|---|---|---|']
  for task,peak,inp,limit in measurements:
   out.append('|'+ '|'.join([task,f'**{peak:.3f}**' if peak is not None else '미실시',f'{peak/RAM*100:.2f}%' if peak is not None else '—',inp,limit])+'|')
  out+=['','메모리는PyTorch의경우CUDA allocated peak다. reserved는예약량으로별도다. **AlphaFold2는컴파일·warmup등을포함한JAX프로세스누적peak**이므로CUDA구간peak와같은정의로비교하지않는다. RSS와CUDA를더하지않는다. 표는시험한조건의관측값이며모델의절대최대나모든모델의80%달성을의미하지않는다.','','## 2. 기본 정보·입력 검사·추론·학습·평가','']
  out.append(shifted((R/parts[0]).read_text()))
  for idx,part in enumerate(parts[1:],start=3):
   out+=['',f'## {idx}. 후속 실험 통합 — '+{'scGPT_memory_batch.md':'학습 배치 확대','scGPT_realdata_full.md':'실제 전체세포 학습','scGPT_actual_300k.md':'실제30만세포 추론','Cellpose_brain_NucMM.md':'뇌 조직 2D·3D 검증'}.get(part,'배치 확대 및 자원 검증'),'',shifted((R/part).read_text())]
  out+=['','## 기록 해석','','본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.']
  (R/'FINAL.md').write_text('\n'.join(out)+'\n');infer=[x[1] for x in measurements if ('추론' in x[0] or '관측값' in x[0]) and x[1] is not None];learn=[x[1] for x in measurements if not ('추론' in x[0] or '관측값' in x[0]) and x[1] is not None];entries.append((title,max(infer) if infer else None,max(learn) if learn else None,'학습용량probe만 확인' if model=='scgpt' else '미확인',model))
 out=['# 최종 통합 보고 — 모델별 입력·추론·학습 메모리','','**이 표에서 모델을 고른 뒤 최종 보고서 하나만 읽으면 된다. 각 문서에 기본 조사·입력 형태·추론·학습·배치 후속·평가·검증을 모두 합쳤다.**','','## 먼저 알아야 할 결론','','메모리80% 목표는allocated약97.35GiB다. 현재이조건을입증한것은scGPT의FP32/1200자리padding/학습용량probe(batch284약80.54%,batch292약82.79%)뿐이다. 일반적인전체세포파인튜닝에서같이달성했다는뜻은아니다. **나머지모델의80%조건은미확인**이다. 후속run을실행했다는것과80%목표를해결했다는것을구분한다.','','## 모델별 최종 보고서','','|모델|추론 관측최대 GiB|학습 관측최대 GiB|메모리80% 조건|읽을 파일|','|---|---|---|---|---|']
 for title,i,t,status,model in entries:out.append('|'+ '|'.join([title,f'{i:.3f}' if i is not None else '미실시',f'{t:.3f}' if t is not None else '미실시',status,f'[최종 보고서]({model}/reports/FINAL.md)'])+'|')
 out+=['','학습열에는작업방식이서로다른scVI처음부터학습·ESM부분파인튜닝·RNA-FM선형head학습·scGPT용량probe가함께있다. 숫자만으로학습메모리순위를매기지말고각최종보고서첫표에서**무엇을학습했는지·입력·배치**를확인한다. AF2메모리는JAX프로세스누적peak라PyTorch실행구간allocated와정의가다르다.','','원본증거를더확인할때만최종보고서안의run JSON/CSV링크를이용하면된다. 이전기본/후속파일은추적용으로보존하되주요읽기경로는이문서와각모델FINAL.md다.']
 (target/'FINAL_REPORT.md').write_text('\n'.join(out)+'\n')
 print('INTEGRATED',target,len(entries))
# Copy the legacy DNA evidence into publication as well.
dst=PUB/'dnabert2/results/legacy_training_4000bp';dst.mkdir(exist_ok=True)
for p in (ROOT/'dnabert2/results/legacy_training_4000bp').glob('*.json'):shutil.copy2(p,dst/p.name)

# Markdown에서 실행 파일·설정·측정 기록 복원

GitHub 최신 트리에는 Markdown 보고서와 Git 관리 파일만 둔다. 실행 코드 전문은각모델reports/FINAL.md, 공통실행도구는docs/COMMON_EXECUTION_CODE.md, JSON/CSV/환경텍스트/그림원본은해당폴더의RECORDS.md에담았다.

아래코드는실제로검증에사용한복원도구전문이다. 저장소루트에서실행하며기본동작은**해시확인만**한다. `restore.py`라는로컬파일로저장해 `python3 restore.py`로검증하고, `python3 restore.py --write`로원래상대경로에파일을복원할수있다. 실제모델실험을자동실행하지않으며기존의다른내용을덮어쓰지않는다. 복원파일은.gitignore에의해Git에추가되지않는다.

이도구의SHA256: `1d51dcddd709c0a17818fec4491c3b6d01076269e5ae98bd1c08927626861ab6`

```python
from pathlib import Path
import argparse,base64,hashlib,json,re,sys
p=argparse.ArgumentParser(description='Verify or restore original files embedded in Markdown. No benchmark is executed.');p.add_argument('--write',action='store_true',help='restore verified original files under repository root');args=p.parse_args();root=Path.cwd().resolve();files={};aliases=[];blocks=0
for md in root.rglob('*.md'):
 if '.git' in md.parts:continue
 lines=md.read_text(encoding='utf-8').splitlines(keepends=True);i=0
 while i<len(lines):
  line=lines[i].strip()
  if line.startswith('<!-- RESTORE_ALIAS ') and line.endswith(' -->'):aliases.append(json.loads(line[len('<!-- RESTORE_ALIAS '):-len(' -->')]))
  if not line.startswith('<!-- RESTORE_FILE ') or not line.endswith(' -->'):i+=1;continue
  meta=json.loads(line[len('<!-- RESTORE_FILE '):-len(' -->')]);i+=1;opening=re.match(r'^(`{3,})[^\n]*\n?$',lines[i]);assert opening,(md,i);fence=opening.group(1);i+=1;content=[]
  while i<len(lines) and lines[i].strip()!=fence:content.append(lines[i]);i+=1
  assert i<len(lines),(md,'missing closing fence');payload=''.join(content)
  if meta['encoding']=='base64':data=base64.b64decode(payload.strip(),validate=True)
  else:
   if meta.get('added_final_newline'):assert payload.endswith('\n');payload=payload[:-1]
   data=payload.encode('utf-8')
  actual=hashlib.sha256(data).hexdigest();assert actual==meta['sha256'],(md,meta['path'],'SHA256 mismatch');path=(root/meta['path']).resolve();assert root in path.parents,(meta['path'],'path outside repository');previous=files.get(meta['path']);assert previous is None or previous['data']==data,(meta['path'],'conflicting code versions');files[meta['path']]={'data':data,'sha256':actual};blocks+=1;i+=1
pending=aliases[:]
while pending:
 progress=False
 for meta in pending[:]:
  if meta['source'] not in files:continue
  src=files[meta['source']];assert src['sha256']==meta['sha256'];previous=files.get(meta['path']);assert previous is None or previous['data']==src['data'];files[meta['path']]=src;pending.remove(meta);progress=True
 assert progress,('unresolved aliases',pending)
if args.write:
 for name,item in files.items():
  dest=(root/name).resolve();assert root in dest.parents;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and dest.read_bytes()!=item['data']:raise RuntimeError('Refusing to overwrite different existing file: '+name)
  dest.write_bytes(item['data'])
print(json.dumps({'verified_embedded_blocks':blocks,'restorable_original_paths':len(files),'restored':args.write},ensure_ascii=False))
```

일부입력원본·모델가중치·대형배열·가상환경은애초에Git에포함하지않았다. 해당자료는보고서출처·해시로따로확보한다. 그림은이전고정커밋의원본링크로표시하고,오프라인재현용PNGbytes도RECORDS.md의base64블록에보존한다.

실행당시snapshot이없는현재소스를당시와동일하다고단정하지않는다. 각코드의경로·원문해시·근거상태를같이읽는다. 기존커밋이력은강제로삭제/재작성하지않았으며최신버전에서독립실행파일을제거했다.

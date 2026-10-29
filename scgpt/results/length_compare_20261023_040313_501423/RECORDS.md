# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-d50c252c8230

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/comparison.csv` · SHA256: `bedb062478bb520a6c779660686c281f262c1c03bea50c653a648b9b73de88ba`

<details>
<summary>comparison.csv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/comparison.csv", "sha256": "bedb062478bb520a6c779660686c281f262c1c03bea50c653a648b9b73de88ba", "encoding": "utf-8", "added_final_newline": false} -->
```csv
task,source_split,cells,max_length_including_cls,actual_tokens_min,actual_tokens_mean,actual_tokens_max,batch_padded_tokens_mean,batch_padded_tokens_max,time_seconds,cells_per_second,gpu_peak_allocated_gib,gpu_peak_reserved_gib,process_rss_peak_observed_gib,gpu_allocation_pct_of_system_ram,system_total_memory_gib,whole_process_seconds,same_cells_and_order_verified,checkpoint_and_input_hashes_verified,saved_model_reload_verified,model_bundle,timing_scope,results_directory
finetune,train,1000,256,72,220.004,256,256.0,256,9.221140660112724,108.44645330329182,1.2221388816833496,1.45703125,2.6563873291015625,1.0043256114518113,121.68751525878906,115.47962304390967,True,True,True,/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001,"tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization",/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L0256
finetune,train,1000,512,72,286.681,512,474.232,512,15.796376958023757,63.30565563593054,1.8074431419372559,2.146484375,2.658599853515625,1.4853151846296004,121.68751525878906,121.79608216392808,True,True,True,/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001,"tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization",/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L0512
finetune,train,1000,1024,72,309.995,1024,612.536,1024,21.328996536089107,46.88453103304598,2.9719724655151367,3.630859375,2.7067222595214844,2.4422985868309786,121.68751525878906,123.01876717992127,True,True,True,/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001,"tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization",/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L1024
finetune,train,1000,1200,72,310.553,1200,615.896,1200,21.51013845205307,46.48970541166151,3.374143600463867,4.5546875,2.7139358520507812,2.7727935715411567,121.68751525878906,122.77139263506979,True,True,True,/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001,"tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization",/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L1200
inference,test,1000,256,185,255.667,256,256.0,256,1.2901826270390302,775.0840687531195,0.25962162017822266,0.28125,2.0585098266601562,0.2133510735477616,121.68751525878906,101.44052228494547,True,True,,,"gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording",/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L0256
inference,test,1000,512,185,459.925,512,504.192,512,2.4339966219849885,410.8469136594255,0.29481029510498047,0.3125,1.7441253662109375,0.24226831690828476,121.68751525878906,105.35337310284376,True,True,,,"gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording",/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L0512
inference,test,1000,1024,185,610.885,1024,796.016,1024,3.531116613186896,283.1965379635203,0.3615903854370117,0.41796875,2.047393798828125,0.29714665852781086,121.68751525878906,98.20881273387931,True,True,,,"gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording",/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L1024
inference,test,1000,1200,185,616.004,1200,825.56,1200,3.588488161796704,278.6688864257851,0.36938047409057617,0.51171875,1.7204704284667969,0.3035483741327335,121.68751525878906,107.93387867789716,True,True,,,"gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording",/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L1200
```
<!-- END_RESTORE_FILE -->


</details>

## file-468f1acdd77c

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/comparison.tsv` · SHA256: `f25e04f03d24b391c24bec08a6dc4e8d52398c59d84a1ae37ae6f33258e9138c`

<details>
<summary>comparison.tsv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/comparison.tsv", "sha256": "f25e04f03d24b391c24bec08a6dc4e8d52398c59d84a1ae37ae6f33258e9138c", "encoding": "utf-8", "added_final_newline": false} -->
```tsv
task	source_split	cells	max_length_including_cls	actual_tokens_min	actual_tokens_mean	actual_tokens_max	batch_padded_tokens_mean	batch_padded_tokens_max	time_seconds	cells_per_second	gpu_peak_allocated_gib	gpu_peak_reserved_gib	process_rss_peak_observed_gib	gpu_allocation_pct_of_system_ram	system_total_memory_gib	whole_process_seconds	same_cells_and_order_verified	checkpoint_and_input_hashes_verified	saved_model_reload_verified	model_bundle	timing_scope	results_directory
finetune	train	1000	256	72	220.004	256	256.0	256	9.221140660112724	108.44645330329182	1.2221388816833496	1.45703125	2.6563873291015625	1.0043256114518113	121.68751525878906	115.47962304390967	True	True	True	/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001	tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization	/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L0256
finetune	train	1000	512	72	286.681	512	474.232	512	15.796376958023757	63.30565563593054	1.8074431419372559	2.146484375	2.658599853515625	1.4853151846296004	121.68751525878906	121.79608216392808	True	True	True	/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001	tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization	/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L0512
finetune	train	1000	1024	72	309.995	1024	612.536	1024	21.328996536089107	46.88453103304598	2.9719724655151367	3.630859375	2.7067222595214844	2.4422985868309786	121.68751525878906	123.01876717992127	True	True	True	/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001	tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization	/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L1024
finetune	train	1000	1200	72	310.553	1200	615.896	1200	21.51013845205307	46.48970541166151	3.374143600463867	4.5546875	2.7139358520507812	2.7727935715411567	121.68751525878906	122.77139263506979	True	True	True	/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001	tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization	/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L1200
inference	test	1000	256	185	255.667	256	256.0	256	1.2901826270390302	775.0840687531195	0.25962162017822266	0.28125	2.0585098266601562	0.2133510735477616	121.68751525878906	101.44052228494547	True	True			gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording	/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L0256
inference	test	1000	512	185	459.925	512	504.192	512	2.4339966219849885	410.8469136594255	0.29481029510498047	0.3125	1.7441253662109375	0.24226831690828476	121.68751525878906	105.35337310284376	True	True			gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording	/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L0512
inference	test	1000	1024	185	610.885	1024	796.016	1024	3.531116613186896	283.1965379635203	0.3615903854370117	0.41796875	2.047393798828125	0.29714665852781086	121.68751525878906	98.20881273387931	True	True			gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording	/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L1024
inference	test	1000	1200	185	616.004	1200	825.56	1200	3.588488161796704	278.6688864257851	0.36938047409057617	0.51171875	1.7204704284667969	0.3035483741327335	121.68751525878906	107.93387867789716	True	True			gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording	/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L1200
```
<!-- END_RESTORE_FILE -->


</details>

## file-2d1c8f2290a2

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/comparison_table.txt` · SHA256: `4d6025e22148acb19f2b0b0d2b4100bdbb9efd6888356fe020db6268ee65adad`

<details>
<summary>comparison_table.txt — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/comparison_table.txt", "sha256": "4d6025e22148acb19f2b0b0d2b4100bdbb9efd6888356fe020db6268ee65adad", "encoding": "utf-8", "added_final_newline": false} -->
```txt
     task  cap  actual_mean  padded_mean  seconds  GPU_alloc_GiB  GPU_res_GiB  RSS_GiB
 finetune  256      220.004      256.000    9.221          1.222        1.457    2.656
 finetune  512      286.681      474.232   15.796          1.807        2.146    2.659
 finetune 1024      309.995      612.536   21.329          2.972        3.631    2.707
 finetune 1200      310.553      615.896   21.510          3.374        4.555    2.714
inference  256      255.667      256.000    1.290          0.260        0.281    2.059
inference  512      459.925      504.192    2.434          0.295        0.312    1.744
inference 1024      610.885      796.016    3.531          0.362        0.418    2.047
inference 1200      616.004      825.560    3.588          0.369        0.512    1.720
```
<!-- END_RESTORE_FILE -->


</details>

## file-e8ca15b4d652

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/container_image.txt` · SHA256: `ed859ddfe22da12dea67d60b1deffc2b5cf0b9ae058e70dce307289fd8b3b85d`

<details>
<summary>container_image.txt — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/container_image.txt", "sha256": "ed859ddfe22da12dea67d60b1deffc2b5cf0b9ae058e70dce307289fd8b3b85d", "encoding": "utf-8", "added_final_newline": false} -->
```txt
Image=nvcr.io/nvidia/pytorch:26.06-py3 ImageID=sha256:9629b436aef8bd90147fd657137047aee94e7b81ada54c3f7209cbce1d24b490
```
<!-- END_RESTORE_FILE -->


</details>

## file-9e2b36fbf6c7

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/experiment_config.json` · SHA256: `33c468aed1f549756d6add2b5648b032872634848dff522f2f381711696a9400`

<details>
<summary>experiment_config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/experiment_config.json", "sha256": "33c468aed1f549756d6add2b5648b032872634848dff522f2f381711696a9400", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "cases": [
    {
      "name": "inference_L0256",
      "task": "inference",
      "max_length": 256
    },
    {
      "name": "inference_L0512",
      "task": "inference",
      "max_length": 512
    },
    {
      "name": "inference_L1024",
      "task": "inference",
      "max_length": 1024
    },
    {
      "name": "inference_L1200",
      "task": "inference",
      "max_length": 1200
    },
    {
      "name": "finetune_L0256",
      "task": "finetune",
      "max_length": 256
    },
    {
      "name": "finetune_L0512",
      "task": "finetune",
      "max_length": 512
    },
    {
      "name": "finetune_L1024",
      "task": "finetune",
      "max_length": 1024
    },
    {
      "name": "finetune_L1200",
      "task": "finetune",
      "max_length": 1200
    }
  ],
  "seed": 42,
  "cells_per_task": 1000,
  "batch_size": 8,
  "precision": "FP32",
  "epochs_per_finetune": 1,
  "repeats_per_condition": 1,
  "length_policy": "official random nonzero-gene sampling with dynamic batch padding; cap includes CLS",
  "inference_split": "test",
  "finetune_split": "train; stratified 20% validation held out",
  "source_sha256": {
    "01_inference.py": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c",
    "02_finetune.py": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc"
  }
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-b9defdd22ec3

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/finetune_input_cells.tsv` · SHA256: `27f9974e7eea2b5e98494d9604ed5cc114b96bc4bdd7c71492db96026bc6372b`

<details>
<summary>finetune_input_cells.tsv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/finetune_input_cells.tsv", "sha256": "27f9974e7eea2b5e98494d9604ed5cc114b96bc4bdd7c71492db96026bc6372b", "encoding": "utf-8", "added_final_newline": false} -->
```tsv
original_row_index	cell_id	Celltype	tokens_before_sampling_including_cls
10264	D30-5_10	alpha	315
7463	human4_lib1.final_cell_0285	ductal	558
2620	human2_lib2.final_cell_0159	alpha	250
2530	human2_lib2.final_cell_0069	PSC	509
8758	D29-2_25	alpha	364
7420	human4_lib1.final_cell_0242	ductal	497
4315	human3_lib1.final_cell_0700	PSC	308
2328	human2_lib1.final_cell_0423	ductal	428
4264	human3_lib1.final_cell_0649	alpha	175
3622	human2_lib3.final_cell_0563	ductal	190
2789	human2_lib2.final_cell_0328	beta	184
5273	human3_lib2.final_cell_0772	alpha	162
2374	human2_lib1.final_cell_0469	delta	121
4199	human3_lib1.final_cell_0584	delta	122
8059	human4_lib3.final_cell_0283	ductal	473
7668	human4_lib1.final_cell_0490	beta	220
3050	human2_lib2.final_cell_0589	beta	101
1665	human1_lib3.final_cell_0488	beta	120
3276	human2_lib3.final_cell_0217	ductal	402
2396	human2_lib1.final_cell_0491	alpha	161
7324	human4_lib1.final_cell_0146	alpha	271
34	human1_lib1.final_cell_0038	beta	279
5792	human3_lib3.final_cell_0423	alpha	231
8715	D29-1_69	alpha	521
5926	human3_lib3.final_cell_0557	alpha	192
3418	human2_lib3.final_cell_0359	alpha	158
3555	human2_lib3.final_cell_0496	mast	162
3487	human2_lib3.final_cell_0428	PSC	383
8067	human4_lib3.final_cell_0291	ductal	432
8467	human4_lib3.final_cell_0691	beta	157
9936	D30-1_29	acinar	622
5306	human3_lib2.final_cell_0805	macrophage	211
2497	human2_lib2.final_cell_0036	ductal	351
9741	D31-6_84	PSC	911
9828	D31-7_92	PP	390
5859	human3_lib3.final_cell_0490	ductal	423
880	human1_lib2.final_cell_0317	beta	176
3999	human3_lib1.final_cell_0384	alpha	194
9584	D31-5_11	alpha	286
3387	human2_lib3.final_cell_0328	ductal	354
5265	human3_lib2.final_cell_0764	alpha	159
10565	D30-8_57	PSC	990
9312	D31-1_88	acinar	854
4498	human3_lib1.final_cell_0883	beta	104
10491	D30-7_72	PSC	763
5359	human3_lib2.final_cell_0858	endothelial	188
9927	D30-1_20	delta	497
3161	human2_lib3.final_cell_0102	PSC	527
8739	D29-2_1	PSC	978
5343	human3_lib2.final_cell_0842	alpha	186
6852	human3_lib4.final_cell_0556	alpha	215
10297	D30-5_48	alpha	240
4464	human3_lib1.final_cell_0849	delta	112
560	human1_lib1.final_cell_0564	endothelial	212
3431	human2_lib3.final_cell_0372	alpha	206
5771	human3_lib3.final_cell_0402	ductal	323
8083	human4_lib3.final_cell_0307	beta	219
1341	human1_lib3.final_cell_0163	acinar	375
7359	human4_lib1.final_cell_0181	ductal	518
2019	human2_lib1.final_cell_0114	alpha	264
2851	human2_lib2.final_cell_0390	alpha	217
10421	D30-6_88	alpha	497
6539	human3_lib4.final_cell_0243	beta	205
414	human1_lib1.final_cell_0418	ductal	278
7019	human3_lib4.final_cell_0723	beta	124
4836	human3_lib2.final_cell_0335	beta	221
4276	human3_lib1.final_cell_0661	acinar	187
6905	human3_lib4.final_cell_0609	ductal	314
5248	human3_lib2.final_cell_0747	alpha	172
3693	human3_lib1.final_cell_0078	acinar	310
2898	human2_lib2.final_cell_0437	alpha	193
4853	human3_lib2.final_cell_0352	beta	256
5349	human3_lib2.final_cell_0848	alpha	120
1879	human1_lib3.final_cell_0702	beta	119
2087	human2_lib1.final_cell_0182	beta	253
2529	human2_lib2.final_cell_0068	beta	309
4747	human3_lib2.final_cell_0246	alpha	210
5016	human3_lib2.final_cell_0515	alpha	127
1751	human1_lib3.final_cell_0574	PSC	281
7117	human3_lib4.final_cell_0821	alpha	133
10529	D30-8_16	beta	658
9206	D29-8_60	alpha	345
9587	D31-5_14	ductal	641
1333	human1_lib3.final_cell_0155	delta	221
2420	human2_lib1.final_cell_0515	beta	170
6076	human3_lib3.final_cell_0707	alpha	155
5275	human3_lib2.final_cell_0774	acinar	183
1772	human1_lib3.final_cell_0595	PSC	269
10170	D30-3_92	alpha	354
3786	human3_lib1.final_cell_0171	alpha	234
845	human1_lib2.final_cell_0282	beta	193
6718	human3_lib4.final_cell_0422	beta	207
742	human1_lib2.final_cell_0179	beta	220
9124	D29-7_45	beta	486
3782	human3_lib1.final_cell_0167	beta	222
7258	human4_lib1.final_cell_0079	ductal	457
1529	human1_lib3.final_cell_0352	beta	262
7983	human4_lib3.final_cell_0207	ductal	469
7111	human3_lib4.final_cell_0815	beta	130
4465	human3_lib1.final_cell_0850	beta	96
4850	human3_lib2.final_cell_0349	alpha	236
4905	human3_lib2.final_cell_0404	alpha	265
9907	D31-8_88	beta	567
2385	human2_lib1.final_cell_0480	alpha	223
6332	human3_lib4.final_cell_0036	acinar	397
7454	human4_lib1.final_cell_0276	beta	264
1171	human1_lib2.final_cell_0608	alpha	126
10424	D30-6_91	alpha	541
503	human1_lib1.final_cell_0507	PSC	207
1176	human1_lib2.final_cell_0613	alpha	104
1362	human1_lib3.final_cell_0184	ductal	509
8419	human4_lib3.final_cell_0643	PP	144
4516	human3_lib2.final_cell_0009	acinar	450
7517	human4_lib1.final_cell_0339	beta	332
7060	human3_lib4.final_cell_0764	beta	145
4327	human3_lib1.final_cell_0712	alpha	199
1632	human1_lib3.final_cell_0455	beta	190
1289	human1_lib3.final_cell_0110	delta	251
255	human1_lib1.final_cell_0259	beta	211
6505	human3_lib4.final_cell_0209	alpha	169
942	human1_lib2.final_cell_0379	PP	163
3235	human2_lib3.final_cell_0176	alpha	139
3775	human3_lib1.final_cell_0160	ductal	429
2912	human2_lib2.final_cell_0451	alpha	152
9302	D31-1_78	alpha	438
3769	human3_lib1.final_cell_0154	beta	270
4084	human3_lib1.final_cell_0469	ductal	321
8600	D28-3_73	ductal	1123
5085	human3_lib2.final_cell_0584	alpha	216
8842	D29-3_34	alpha	350
1149	human1_lib2.final_cell_0586	PSC	209
1593	human1_lib3.final_cell_0416	beta	208
1615	human1_lib3.final_cell_0438	ductal	333
2023	human2_lib1.final_cell_0118	alpha	265
5597	human3_lib3.final_cell_0227	acinar	410
315	human1_lib1.final_cell_0319	endothelial	386
10072	D30-2_85	alpha	590
3590	human2_lib3.final_cell_0531	alpha	124
8370	human4_lib3.final_cell_0594	ductal	356
6459	human3_lib4.final_cell_0163	beta	207
2002	human2_lib1.final_cell_0097	alpha	169
7972	human4_lib3.final_cell_0196	PP	228
4964	human3_lib2.final_cell_0463	alpha	200
9973	D30-1_72	acinar	771
2127	human2_lib1.final_cell_0222	alpha	190
1721	human1_lib3.final_cell_0544	endothelial	250
3312	human2_lib3.final_cell_0253	beta	220
6305	human3_lib4.final_cell_0003	acinar	384
5337	human3_lib2.final_cell_0836	delta	118
4060	human3_lib1.final_cell_0445	alpha	204
7354	human4_lib1.final_cell_0176	delta	283
5363	human3_lib2.final_cell_0862	beta	97
6764	human3_lib4.final_cell_0468	ductal	336
8374	human4_lib3.final_cell_0598	alpha	143
8798	D29-2_71	alpha	556
5330	human3_lib2.final_cell_0829	beta	106
2501	human2_lib2.final_cell_0040	beta	220
5867	human3_lib3.final_cell_0498	alpha	189
1420	human1_lib3.final_cell_0243	PSC	514
6422	human3_lib4.final_cell_0126	acinar	279
7882	human4_lib3.final_cell_0106	ductal	524
6091	human3_lib3.final_cell_0722	ductal	346
3842	human3_lib1.final_cell_0227	alpha	232
10364	D30-6_28	PSC	1106
8475	human4_lib3.final_cell_0699	beta	108
10543	D30-8_32	delta	627
724	human1_lib2.final_cell_0161	beta	167
7598	human4_lib1.final_cell_0420	ductal	419
10047	D30-2_56	acinar	947
5297	human3_lib2.final_cell_0796	PSC	233
9854	D31-8_26	PP	333
5576	human3_lib3.final_cell_0206	beta	252
3720	human3_lib1.final_cell_0105	acinar	405
7523	human4_lib1.final_cell_0345	alpha	248
10518	D30-8_5	alpha	487
8557	D28-2_89	acinar	826
7613	human4_lib1.final_cell_0435	alpha	212
8152	human4_lib3.final_cell_0376	PSC	409
9845	D31-8_15	PP	302
8305	human4_lib3.final_cell_0529	PP	172
5407	human3_lib3.final_cell_0036	acinar	419
7659	human4_lib1.final_cell_0481	alpha	201
7278	human4_lib1.final_cell_0100	beta	315
6532	human3_lib4.final_cell_0236	beta	261
5405	human3_lib3.final_cell_0034	delta	200
3690	human3_lib1.final_cell_0075	delta	202
4104	human3_lib1.final_cell_0489	macrophage	358
6771	human3_lib4.final_cell_0475	beta	194
7848	human4_lib3.final_cell_0072	PSC	561
8603	D28-3_77	acinar	751
5244	human3_lib2.final_cell_0743	ductal	268
9062	D29-6_53	alpha	411
18	human1_lib1.final_cell_0022	alpha	232
480	human1_lib1.final_cell_0484	PP	122
438	human1_lib1.final_cell_0442	beta	149
5936	human3_lib3.final_cell_0567	alpha	208
2904	human2_lib2.final_cell_0443	ductal	351
648	human1_lib2.final_cell_0085	beta	221
6362	human3_lib4.final_cell_0066	acinar	349
4890	human3_lib2.final_cell_0389	alpha	255
5546	human3_lib3.final_cell_0176	acinar	320
9148	D29-7_72	alpha	459
5319	human3_lib2.final_cell_0818	acinar	191
1501	human1_lib3.final_cell_0324	beta	133
2683	human2_lib2.final_cell_0222	alpha	232
4288	human3_lib1.final_cell_0673	beta	148
8723	D29-1_77	beta	432
8108	human4_lib3.final_cell_0332	alpha	242
7963	human4_lib3.final_cell_0187	PP	246
725	human1_lib2.final_cell_0162	beta	221
2949	human2_lib2.final_cell_0488	PP	148
6940	human3_lib4.final_cell_0644	beta	107
10000	D30-2_8	PP	453
6853	human3_lib4.final_cell_0557	alpha	172
40	human1_lib1.final_cell_0044	PSC	544
9751	D31-6_94	alpha	399
6277	human3_lib3.final_cell_0908	macrophage	215
9273	D31-1_47	beta	527
1210	human1_lib3.final_cell_0024	beta	290
2051	human2_lib1.final_cell_0146	alpha	237
2749	human2_lib2.final_cell_0288	alpha	134
5370	human3_lib2.final_cell_0869	endothelial	188
364	human1_lib1.final_cell_0368	beta	183
1844	human1_lib3.final_cell_0667	beta	98
9755	D31-7_4	endothelial	623
10269	D30-5_15	PSC	661
7518	human4_lib1.final_cell_0340	beta	287
458	human1_lib1.final_cell_0462	beta	138
5328	human3_lib2.final_cell_0827	beta	126
1372	human1_lib3.final_cell_0195	delta	186
402	human1_lib1.final_cell_0406	alpha	164
1009	human1_lib2.final_cell_0446	beta	176
4763	human3_lib2.final_cell_0262	ductal	523
2710	human2_lib2.final_cell_0249	alpha	199
5209	human3_lib2.final_cell_0708	beta	160
7053	human3_lib4.final_cell_0757	ductal	303
8302	human4_lib3.final_cell_0526	alpha	161
4936	human3_lib2.final_cell_0435	beta	204
9829	D31-7_93	alpha	544
9140	D29-7_62	ductal	294
7328	human4_lib1.final_cell_0150	beta	335
7700	human4_lib1.final_cell_0522	beta	164
331	human1_lib1.final_cell_0335	beta	215
1380	human1_lib3.final_cell_0203	alpha	263
3297	human2_lib3.final_cell_0238	ductal	444
7201	human4_lib1.final_cell_0011	alpha	331
6906	human3_lib4.final_cell_0610	acinar	213
581	human1_lib2.final_cell_0018	acinar	384
1276	human1_lib3.final_cell_0096	delta	255
6405	human3_lib4.final_cell_0109	acinar	334
694	human1_lib2.final_cell_0131	alpha	227
2663	human2_lib2.final_cell_0202	alpha	203
894	human1_lib2.final_cell_0331	alpha	204
8091	human4_lib3.final_cell_0315	beta	191
8731	D29-1_86	PP	409
10148	D30-3_68	alpha	494
9172	D29-8_10	delta	378
4912	human3_lib2.final_cell_0411	acinar	228
10391	D30-6_57	ductal	788
4245	human3_lib1.final_cell_0630	ductal	207
633	human1_lib2.final_cell_0070	beta	244
6115	human3_lib3.final_cell_0746	alpha	159
2326	human2_lib1.final_cell_0421	beta	191
6602	human3_lib4.final_cell_0306	beta	133
728	human1_lib2.final_cell_0165	beta	208
3994	human3_lib1.final_cell_0379	delta	219
5779	human3_lib3.final_cell_0410	alpha	242
4616	human3_lib2.final_cell_0115	acinar	420
5993	human3_lib3.final_cell_0624	alpha	179
9147	D29-7_71	epsilon	467
8384	human4_lib3.final_cell_0608	alpha	177
1402	human1_lib3.final_cell_0225	beta	244
10092	D30-3_11	alpha	387
6084	human3_lib3.final_cell_0715	alpha	207
6549	human3_lib4.final_cell_0253	alpha	203
7872	human4_lib3.final_cell_0096	ductal	443
7419	human4_lib1.final_cell_0241	beta	322
8730	D29-1_84	beta	484
3647	human3_lib1.final_cell_0030	delta	210
2849	human2_lib2.final_cell_0388	ductal	361
689	human1_lib2.final_cell_0126	acinar	230
3997	human3_lib1.final_cell_0382	ductal	400
3638	human3_lib1.final_cell_0021	acinar	421
9890	D31-8_69	delta	464
7938	human4_lib3.final_cell_0162	beta	270
9418	D31-3_19	alpha	266
9055	D29-6_43	beta	579
4520	human3_lib2.final_cell_0013	acinar	345
8403	human4_lib3.final_cell_0627	beta	183
1114	human1_lib2.final_cell_0551	beta	152
6470	human3_lib4.final_cell_0174	ductal	370
7035	human3_lib4.final_cell_0739	beta	118
4864	human3_lib2.final_cell_0363	alpha	209
9364	D31-2_52	beta	604
2577	human2_lib2.final_cell_0116	alpha	198
6072	human3_lib3.final_cell_0703	acinar	183
5719	human3_lib3.final_cell_0350	beta	206
9109	D29-7_30	PP	403
4668	human3_lib2.final_cell_0167	PP	207
61	human1_lib1.final_cell_0065	ductal	422
4706	human3_lib2.final_cell_0205	alpha	275
5409	human3_lib3.final_cell_0038	acinar	347
4600	human3_lib2.final_cell_0099	acinar	371
9165	D29-7_92	alpha	489
229	human1_lib1.final_cell_0233	delta	166
2504	human2_lib2.final_cell_0043	ductal	417
1239	human1_lib3.final_cell_0059	beta	300
5261	human3_lib2.final_cell_0760	PSC	236
10445	D30-7_18	alpha	556
7749	human4_lib1.final_cell_0571	delta	152
6268	human3_lib3.final_cell_0899	beta	112
6358	human3_lib4.final_cell_0062	acinar	316
2926	human2_lib2.final_cell_0465	beta	210
4988	human3_lib2.final_cell_0487	alpha	209
7159	human3_lib4.final_cell_0863	delta	72
6784	human3_lib4.final_cell_0488	alpha	192
9385	D31-2_73	ductal	947
7330	human4_lib1.final_cell_0152	alpha	234
1596	human1_lib3.final_cell_0419	delta	153
5267	human3_lib2.final_cell_0766	endothelial	244
1503	human1_lib3.final_cell_0326	PSC	368
2692	human2_lib2.final_cell_0231	beta	221
7195	human3_lib4.final_cell_0899	alpha	96
8360	human4_lib3.final_cell_0584	beta	177
9582	D31-5_9	beta	369
6727	human3_lib4.final_cell_0431	alpha	185
3686	human3_lib1.final_cell_0071	endothelial	587
6479	human3_lib4.final_cell_0183	delta	143
6935	human3_lib4.final_cell_0639	alpha	156
4005	human3_lib1.final_cell_0390	alpha	189
3619	human2_lib3.final_cell_0560	beta	93
6133	human3_lib3.final_cell_0764	alpha	148
9964	D30-1_61	acinar	792
266	human1_lib1.final_cell_0270	alpha	214
1666	human1_lib3.final_cell_0489	alpha	246
9744	D31-6_87	beta	592
10243	D30-4_78	PSC	931
8373	human4_lib3.final_cell_0597	beta	207
9851	D31-8_23	ductal	881
5728	human3_lib3.final_cell_0359	beta	198
3325	human2_lib3.final_cell_0266	beta	114
9743	D31-6_86	acinar	810
3211	human2_lib3.final_cell_0152	alpha	198
1924	human2_lib1.final_cell_0007	delta	226
1044	human1_lib2.final_cell_0481	alpha	136
9753	D31-7_2	alpha	341
2011	human2_lib1.final_cell_0106	alpha	230
3815	human3_lib1.final_cell_0200	acinar	234
10566	D30-8_59	beta	690
4399	human3_lib1.final_cell_0784	PSC	229
5620	human3_lib3.final_cell_0250	ductal	421
2918	human2_lib2.final_cell_0457	alpha	181
4260	human3_lib1.final_cell_0645	alpha	186
2339	human2_lib1.final_cell_0434	ductal	410
5937	human3_lib3.final_cell_0568	alpha	176
7708	human4_lib1.final_cell_0530	delta	167
2251	human2_lib1.final_cell_0346	ductal	408
6559	human3_lib4.final_cell_0263	beta	172
970	human1_lib2.final_cell_0407	delta	156
2150	human2_lib1.final_cell_0245	alpha	190
5441	human3_lib3.final_cell_0071	acinar	377
10374	D30-6_39	alpha	464
4915	human3_lib2.final_cell_0414	ductal	415
9197	D29-8_48	PP	236
7690	human4_lib1.final_cell_0512	beta	212
8550	D28-2_75	acinar	1049
3368	human2_lib3.final_cell_0309	alpha	225
10580	D30-8_75	alpha	481
5797	human3_lib3.final_cell_0428	beta	199
3030	human2_lib2.final_cell_0569	beta	107
1153	human1_lib2.final_cell_0590	endothelial	208
3948	human3_lib1.final_cell_0333	beta	195
5967	human3_lib3.final_cell_0598	alpha	230
6015	human3_lib3.final_cell_0646	acinar	199
9822	D31-7_81	delta	286
10318	D30-5_72	PSC	801
9015	D29-5_73	acinar	367
1604	human1_lib3.final_cell_0427	acinar	188
752	human1_lib2.final_cell_0189	beta	166
6028	human3_lib3.final_cell_0659	alpha	195
174	human1_lib1.final_cell_0178	delta	176
2936	human2_lib2.final_cell_0475	beta	183
3770	human3_lib1.final_cell_0155	acinar	408
9024	D29-5_87	alpha	617
1491	human1_lib3.final_cell_0314	beta	314
10284	D30-5_34	ductal	842
5798	human3_lib3.final_cell_0429	beta	203
5377	human3_lib3.final_cell_0002	acinar	507
4129	human3_lib1.final_cell_0514	alpha	158
6356	human3_lib4.final_cell_0060	acinar	306
4589	human3_lib2.final_cell_0088	acinar	410
5430	human3_lib3.final_cell_0060	acinar	413
7573	human4_lib1.final_cell_0395	PP	229
3377	human2_lib3.final_cell_0318	ductal	480
2352	human2_lib1.final_cell_0447	alpha	186
5997	human3_lib3.final_cell_0628	ductal	415
7982	human4_lib3.final_cell_0206	PSC	545
6497	human3_lib4.final_cell_0201	beta	196
2187	human2_lib1.final_cell_0282	beta	229
9560	D31-4_77	ductal	1062
10220	D30-4_50	alpha	616
7028	human3_lib4.final_cell_0732	acinar	183
593	human1_lib2.final_cell_0030	beta	263
8814	D29-2_92	PP	257
4340	human3_lib1.final_cell_0725	endothelial	297
7658	human4_lib1.final_cell_0480	alpha	218
9283	D31-1_58	acinar	758
2422	human2_lib1.final_cell_0517	ductal	196
2612	human2_lib2.final_cell_0151	delta	158
8502	D28-1_65	beta	703
8489	D28-1_38	acinar	1244
9799	D31-7_57	ductal	835
8913	D29-4_35	alpha	338
2086	human2_lib1.final_cell_0181	ductal	485
3845	human3_lib1.final_cell_0230	acinar	239
10417	D30-6_84	PSC	1162
2942	human2_lib2.final_cell_0481	alpha	138
3071	human2_lib3.final_cell_0010	delta	254
2358	human2_lib1.final_cell_0453	delta	184
10200	D30-4_29	alpha	390
1589	human1_lib3.final_cell_0412	delta	180
7549	human4_lib1.final_cell_0371	PSC	499
6443	human3_lib4.final_cell_0147	acinar	241
7240	human4_lib1.final_cell_0059	beta	351
1108	human1_lib2.final_cell_0545	beta	86
9848	D31-8_19	ductal	710
1552	human1_lib3.final_cell_0375	beta	231
635	human1_lib2.final_cell_0072	acinar	334
5791	human3_lib3.final_cell_0422	beta	223
5538	human3_lib3.final_cell_0168	acinar	293
1450	human1_lib3.final_cell_0273	alpha	303
7076	human3_lib4.final_cell_0780	beta	72
7893	human4_lib3.final_cell_0117	beta	254
1222	human1_lib3.final_cell_0037	beta	331
1517	human1_lib3.final_cell_0340	PP	215
6101	human3_lib3.final_cell_0732	alpha	174
1466	human1_lib3.final_cell_0289	alpha	272
10108	D30-3_27	alpha	353
7460	human4_lib1.final_cell_0282	beta	326
228	human1_lib1.final_cell_0232	delta	167
4877	human3_lib2.final_cell_0376	alpha	253
7193	human3_lib4.final_cell_0897	alpha	92
3569	human2_lib3.final_cell_0510	ductal	271
4238	human3_lib1.final_cell_0623	delta	157
9307	D31-1_83	ductal	638
6216	human3_lib3.final_cell_0847	beta	116
9404	D31-2_94	ductal	622
2516	human2_lib2.final_cell_0055	alpha	315
2865	human2_lib2.final_cell_0404	alpha	204
10420	D30-6_87	alpha	536
10001	D30-2_9	ductal	865
3132	human2_lib3.final_cell_0073	alpha	229
4153	human3_lib1.final_cell_0538	beta	141
1374	human1_lib3.final_cell_0197	ductal	443
1356	human1_lib3.final_cell_0178	delta	228
4495	human3_lib1.final_cell_0880	beta	104
6528	human3_lib4.final_cell_0232	ductal	386
4686	human3_lib2.final_cell_0185	delta	203
7261	human4_lib1.final_cell_0083	delta	252
6436	human3_lib4.final_cell_0140	beta	204
1738	human1_lib3.final_cell_0561	beta	153
2517	human2_lib2.final_cell_0056	beta	257
5679	human3_lib3.final_cell_0310	beta	227
2278	human2_lib1.final_cell_0373	ductal	377
797	human1_lib2.final_cell_0234	beta	170
5236	human3_lib2.final_cell_0735	beta	162
932	human1_lib2.final_cell_0369	beta	164
9568	D31-4_85	beta	553
6974	human3_lib4.final_cell_0678	alpha	175
7827	human4_lib3.final_cell_0051	beta	285
5563	human3_lib3.final_cell_0193	beta	246
4943	human3_lib2.final_cell_0442	alpha	223
4594	human3_lib2.final_cell_0093	acinar	361
6933	human3_lib4.final_cell_0637	alpha	224
2340	human2_lib1.final_cell_0435	delta	114
793	human1_lib2.final_cell_0230	delta	159
116	human1_lib1.final_cell_0120	beta	249
79	human1_lib1.final_cell_0083	beta	253
1095	human1_lib2.final_cell_0532	ductal	286
7590	human4_lib1.final_cell_0412	beta	242
8718	D29-1_72	PP	474
3163	human2_lib3.final_cell_0104	alpha	214
270	human1_lib1.final_cell_0274	alpha	196
10449	D30-7_23	delta	626
10096	D30-3_15	PP	378
8621	D28-4_26	PP	451
7095	human3_lib4.final_cell_0799	alpha	157
8310	human4_lib3.final_cell_0534	beta	228
337	human1_lib1.final_cell_0341	beta	110
82	human1_lib1.final_cell_0086	delta	244
2190	human2_lib1.final_cell_0285	beta	275
5558	human3_lib3.final_cell_0188	acinar	327
5710	human3_lib3.final_cell_0341	ductal	343
8979	D29-5_31	alpha	288
9199	D29-8_50	beta	393
4602	human3_lib2.final_cell_0101	acinar	435
8948	D29-4_77	alpha	320
253	human1_lib1.final_cell_0257	alpha	161
2956	human2_lib2.final_cell_0495	alpha	185
5803	human3_lib3.final_cell_0434	alpha	180
6141	human3_lib3.final_cell_0772	alpha	155
224	human1_lib1.final_cell_0228	beta	192
7563	human4_lib1.final_cell_0385	beta	231
9937	D30-1_30	acinar	723
5284	human3_lib2.final_cell_0783	alpha	158
7384	human4_lib1.final_cell_0206	beta	285
1115	human1_lib2.final_cell_0552	mast	140
7576	human4_lib1.final_cell_0398	beta	250
5336	human3_lib2.final_cell_0835	alpha	109
3888	human3_lib1.final_cell_0273	alpha	184
9608	D31-5_37	ductal	854
8929	D29-4_55	alpha	574
940	human1_lib2.final_cell_0377	acinar	238
1661	human1_lib3.final_cell_0484	PSC	356
2527	human2_lib2.final_cell_0066	beta	214
10539	D30-8_27	beta	286
1232	human1_lib3.final_cell_0050	delta	255
6911	human3_lib4.final_cell_0615	acinar	188
1590	human1_lib3.final_cell_0413	beta	183
5140	human3_lib2.final_cell_0639	alpha	191
4986	human3_lib2.final_cell_0485	alpha	267
7464	human4_lib1.final_cell_0286	alpha	242
1026	human1_lib2.final_cell_0463	delta	151
6383	human3_lib4.final_cell_0087	acinar	328
2495	human2_lib2.final_cell_0034	delta	214
10053	D30-2_62	alpha	622
89	human1_lib1.final_cell_0093	beta	252
1785	human1_lib3.final_cell_0608	beta	159
736	human1_lib2.final_cell_0173	beta	189
8970	D29-5_17	alpha	377
9529	D31-4_46	delta	588
6281	human3_lib3.final_cell_0912	beta	107
7702	human4_lib1.final_cell_0524	beta	219
7398	human4_lib1.final_cell_0220	ductal	555
3432	human2_lib3.final_cell_0373	alpha	210
978	human1_lib2.final_cell_0415	beta	182
4951	human3_lib2.final_cell_0450	beta	221
3017	human2_lib2.final_cell_0556	delta	149
1762	human1_lib3.final_cell_0585	beta	157
8517	D28-1_90	ductal	992
9819	D31-7_77	acinar	709
3181	human2_lib3.final_cell_0122	alpha	215
10494	D30-7_75	alpha	529
9656	D31-5_87	alpha	619
4330	human3_lib1.final_cell_0715	alpha	204
7741	human4_lib1.final_cell_0563	beta	144
8301	human4_lib3.final_cell_0525	alpha	204
3450	human2_lib3.final_cell_0391	alpha	221
5455	human3_lib3.final_cell_0085	acinar	365
6713	human3_lib4.final_cell_0417	alpha	216
6012	human3_lib3.final_cell_0643	alpha	198
6592	human3_lib4.final_cell_0296	delta	217
6502	human3_lib4.final_cell_0206	alpha	269
3882	human3_lib1.final_cell_0267	alpha	242
7176	human3_lib4.final_cell_0880	alpha	104
5898	human3_lib3.final_cell_0529	PP	181
3737	human3_lib1.final_cell_0122	beta	176
3375	human2_lib3.final_cell_0316	alpha	165
110	human1_lib1.final_cell_0114	beta	234
3741	human3_lib1.final_cell_0126	beta	277
2614	human2_lib2.final_cell_0153	ductal	406
1594	human1_lib3.final_cell_0417	PSC	358
7392	human4_lib1.final_cell_0214	alpha	309
6560	human3_lib4.final_cell_0264	beta	202
4644	human3_lib2.final_cell_0143	ductal	466
5640	human3_lib3.final_cell_0270	beta	218
9814	D31-7_72	beta	435
8581	D28-3_40	acinar	986
8259	human4_lib3.final_cell_0483	beta	211
7435	human4_lib1.final_cell_0257	ductal	446
8251	human4_lib3.final_cell_0475	ductal	386
5702	human3_lib3.final_cell_0333	acinar	267
7959	human4_lib3.final_cell_0183	alpha	234
5958	human3_lib3.final_cell_0589	beta	140
470	human1_lib1.final_cell_0474	beta	121
6162	human3_lib3.final_cell_0793	beta	130
3121	human2_lib3.final_cell_0062	PSC	543
4316	human3_lib1.final_cell_0701	alpha	166
630	human1_lib2.final_cell_0067	acinar	356
7960	human4_lib3.final_cell_0184	alpha	267
3055	human2_lib2.final_cell_0594	beta	119
6621	human3_lib4.final_cell_0325	alpha	219
218	human1_lib1.final_cell_0222	PP	192
7309	human4_lib1.final_cell_0131	ductal	578
8194	human4_lib3.final_cell_0418	ductal	414
5861	human3_lib3.final_cell_0492	alpha	220
6903	human3_lib4.final_cell_0607	alpha	159
9112	D29-7_33	alpha	271
1020	human1_lib2.final_cell_0457	alpha	188
10182	D30-4_11	alpha	534
2662	human2_lib2.final_cell_0201	PSC	479
10427	D30-6_94	beta	511
4710	human3_lib2.final_cell_0209	ductal	475
4233	human3_lib1.final_cell_0618	acinar	169
10123	D30-3_42	alpha	464
8472	human4_lib3.final_cell_0696	beta	142
4460	human3_lib1.final_cell_0845	beta	109
6789	human3_lib4.final_cell_0493	alpha	230
7794	human4_lib3.final_cell_0016	beta	336
1361	human1_lib3.final_cell_0183	ductal	427
6071	human3_lib3.final_cell_0702	PSC	432
825	human1_lib2.final_cell_0262	alpha	198
2561	human2_lib2.final_cell_0100	alpha	246
1734	human1_lib3.final_cell_0557	beta	107
9598	D31-5_25	alpha	241
4292	human3_lib1.final_cell_0677	alpha	147
1541	human1_lib3.final_cell_0364	beta	196
2133	human2_lib1.final_cell_0228	alpha	225
3730	human3_lib1.final_cell_0115	alpha	295
1640	human1_lib3.final_cell_0463	epsilon	242
10271	D30-5_18	delta	605
5218	human3_lib2.final_cell_0717	alpha	198
206	human1_lib1.final_cell_0210	beta	209
1780	human1_lib3.final_cell_0603	ductal	262
7492	human4_lib1.final_cell_0314	alpha	228
4844	human3_lib2.final_cell_0343	ductal	425
3099	human2_lib3.final_cell_0040	alpha	186
8911	D29-4_33	alpha	294
7551	human4_lib1.final_cell_0373	alpha	213
10464	D30-7_41	alpha	791
1396	human1_lib3.final_cell_0219	beta	267
3044	human2_lib2.final_cell_0583	schwann	169
6253	human3_lib3.final_cell_0884	ductal	185
7682	human4_lib1.final_cell_0504	beta	217
6346	human3_lib4.final_cell_0050	delta	182
8838	D29-3_30	alpha	323
1363	human1_lib3.final_cell_0185	alpha	284
4578	human3_lib2.final_cell_0077	beta	233
8835	D29-3_25	alpha	210
3059	human2_lib2.final_cell_0598	alpha	80
3538	human2_lib3.final_cell_0479	beta	153
1446	human1_lib3.final_cell_0269	alpha	292
4574	human3_lib2.final_cell_0073	acinar	342
8239	human4_lib3.final_cell_0463	beta	208
7575	human4_lib1.final_cell_0397	beta	232
9028	D29-6_1	alpha	335
5864	human3_lib3.final_cell_0495	acinar	235
4628	human3_lib2.final_cell_0127	acinar	340
7688	human4_lib1.final_cell_0510	beta	189
9777	D31-7_33	alpha	338
2651	human2_lib2.final_cell_0190	alpha	224
8905	D29-4_26	alpha	321
6806	human3_lib4.final_cell_0510	alpha	220
3725	human3_lib1.final_cell_0110	delta	220
9454	D31-3_61	PP	276
6962	human3_lib4.final_cell_0666	alpha	195
2508	human2_lib2.final_cell_0047	delta	191
4052	human3_lib1.final_cell_0437	ductal	332
2080	human2_lib1.final_cell_0175	alpha	223
1606	human1_lib3.final_cell_0429	endothelial	393
6732	human3_lib4.final_cell_0436	alpha	227
4554	human3_lib2.final_cell_0053	acinar	444
6345	human3_lib4.final_cell_0049	acinar	371
1479	human1_lib3.final_cell_0302	beta	239
2854	human2_lib2.final_cell_0393	ductal	381
9940	D30-1_35	alpha	278
10354	D30-6_18	beta	419
887	human1_lib2.final_cell_0324	PSC	287
7651	human4_lib1.final_cell_0473	alpha	195
7861	human4_lib3.final_cell_0085	beta	305
4824	human3_lib2.final_cell_0323	ductal	496
7847	human4_lib3.final_cell_0071	ductal	470
8857	D29-3_53	alpha	355
3434	human2_lib3.final_cell_0375	alpha	219
360	human1_lib1.final_cell_0364	beta	152
6872	human3_lib4.final_cell_0576	PSC	359
5615	human3_lib3.final_cell_0245	alpha	228
443	human1_lib1.final_cell_0447	beta	104
1349	human1_lib3.final_cell_0171	acinar	370
1988	human2_lib1.final_cell_0083	ductal	503
8755	D29-2_20	beta	532
9949	D30-1_44	endothelial	501
75	human1_lib1.final_cell_0079	beta	233
8013	human4_lib3.final_cell_0237	beta	222
906	human1_lib2.final_cell_0343	macrophage	345
6282	human3_lib3.final_cell_0913	endothelial	209
8866	D29-3_66	alpha	452
5063	human3_lib2.final_cell_0562	beta	198
9688	D31-6_30	alpha	293
9001	D29-5_58	beta	452
10254	D30-4_91	ductal	853
1069	human1_lib2.final_cell_0506	beta	150
3979	human3_lib1.final_cell_0364	beta	220
211	human1_lib1.final_cell_0215	PSC	413
7596	human4_lib1.final_cell_0418	beta	311
4147	human3_lib1.final_cell_0532	beta	187
841	human1_lib2.final_cell_0278	acinar	202
1265	human1_lib3.final_cell_0085	beta	224
10337	D30-5_94	alpha	403
6677	human3_lib4.final_cell_0381	acinar	272
2390	human2_lib1.final_cell_0485	endothelial	351
2546	human2_lib2.final_cell_0085	PSC	517
1481	human1_lib3.final_cell_0304	beta	199
4241	human3_lib1.final_cell_0626	beta	206
83	human1_lib1.final_cell_0087	delta	222
9566	D31-4_83	beta	609
1856	human1_lib3.final_cell_0679	ductal	234
4058	human3_lib1.final_cell_0443	beta	186
2500	human2_lib2.final_cell_0039	alpha	318
7357	human4_lib1.final_cell_0179	PP	314
6730	human3_lib4.final_cell_0434	ductal	360
4206	human3_lib1.final_cell_0591	beta	179
1828	human1_lib3.final_cell_0651	beta	148
4963	human3_lib2.final_cell_0462	acinar	412
2320	human2_lib1.final_cell_0415	alpha	188
2089	human2_lib1.final_cell_0184	alpha	251
1234	human1_lib3.final_cell_0053	beta	301
9222	D29-8_81	delta	298
7990	human4_lib3.final_cell_0214	beta	313
856	human1_lib2.final_cell_0293	endothelial	366
6176	human3_lib3.final_cell_0807	endothelial	261
4113	human3_lib1.final_cell_0498	alpha	201
1245	human1_lib3.final_cell_0065	beta	301
8951	D29-4_81	delta	327
1808	human1_lib3.final_cell_0631	PSC	308
4452	human3_lib1.final_cell_0837	beta	95
7058	human3_lib4.final_cell_0762	acinar	154
7602	human4_lib1.final_cell_0424	alpha	181
6683	human3_lib4.final_cell_0387	beta	161
8633	D28-4_49	acinar	926
848	human1_lib2.final_cell_0285	beta	202
6271	human3_lib3.final_cell_0902	delta	127
7319	human4_lib1.final_cell_0141	delta	226
6681	human3_lib4.final_cell_0385	beta	163
10332	D30-5_89	acinar	685
3881	human3_lib1.final_cell_0266	alpha	197
5755	human3_lib3.final_cell_0386	alpha	245
4013	human3_lib1.final_cell_0398	acinar	263
5557	human3_lib3.final_cell_0187	beta	309
2709	human2_lib2.final_cell_0248	beta	207
8578	D28-3_31	acinar	467
8330	human4_lib3.final_cell_0554	beta	195
3809	human3_lib1.final_cell_0194	delta	184
2563	human2_lib2.final_cell_0102	PSC	509
448	human1_lib1.final_cell_0452	PSC	278
6793	human3_lib4.final_cell_0497	beta	152
7347	human4_lib1.final_cell_0169	delta	275
8858	D29-3_54	alpha	446
2951	human2_lib2.final_cell_0490	alpha	190
2140	human2_lib1.final_cell_0235	alpha	270
1391	human1_lib3.final_cell_0214	beta	264
8248	human4_lib3.final_cell_0472	beta	177
6896	human3_lib4.final_cell_0600	alpha	246
4533	human3_lib2.final_cell_0028	acinar	356
4565	human3_lib2.final_cell_0064	acinar	290
2803	human2_lib2.final_cell_0342	alpha	204
3898	human3_lib1.final_cell_0283	beta	175
4064	human3_lib1.final_cell_0449	beta	218
351	human1_lib1.final_cell_0355	endothelial	353
7027	human3_lib4.final_cell_0731	alpha	214
3183	human2_lib3.final_cell_0124	PSC	484
10460	D30-7_36	alpha	292
4457	human3_lib1.final_cell_0842	PSC	238
7820	human4_lib3.final_cell_0043	ductal	483
2265	human2_lib1.final_cell_0360	alpha	181
10545	D30-8_34	endothelial	622
9114	D29-7_35	PP	378
4424	human3_lib1.final_cell_0809	alpha	152
4728	human3_lib2.final_cell_0227	ductal	448
578	human1_lib2.final_cell_0015	delta	167
7360	human4_lib1.final_cell_0182	alpha	272
5176	human3_lib2.final_cell_0675	acinar	230
7129	human3_lib4.final_cell_0833	acinar	233
9004	D29-5_61	beta	290
5131	human3_lib2.final_cell_0630	ductal	378
5872	human3_lib3.final_cell_0503	beta	138
8232	human4_lib3.final_cell_0456	beta	235
2094	human2_lib1.final_cell_0189	PP	283
4072	human3_lib1.final_cell_0457	beta	178
7049	human3_lib4.final_cell_0753	endothelial	275
8810	D29-2_88	ductal	591
966	human1_lib2.final_cell_0403	alpha	188
10480	D30-7_60	alpha	298
8087	human4_lib3.final_cell_0311	ductal	459
3479	human2_lib3.final_cell_0420	alpha	191
8174	human4_lib3.final_cell_0398	beta	248
7973	human4_lib3.final_cell_0197	beta	344
7707	human4_lib1.final_cell_0529	beta	238
6715	human3_lib4.final_cell_0419	alpha	235
4442	human3_lib1.final_cell_0827	ductal	250
5220	human3_lib2.final_cell_0719	alpha	182
6863	human3_lib4.final_cell_0567	alpha	147
10388	D30-6_54	beta	552
9546	D31-4_63	beta	539
9371	D31-2_59	ductal	666
4165	human3_lib1.final_cell_0550	ductal	326
4849	human3_lib2.final_cell_0348	alpha	280
5485	human3_lib3.final_cell_0115	alpha	278
2771	human2_lib2.final_cell_0310	delta	139
4086	human3_lib1.final_cell_0471	alpha	214
4612	human3_lib2.final_cell_0111	alpha	341
5446	human3_lib3.final_cell_0076	acinar	572
4960	human3_lib2.final_cell_0459	ductal	420
4711	human3_lib2.final_cell_0210	beta	192
3112	human2_lib3.final_cell_0053	beta	222
4647	human3_lib2.final_cell_0146	beta	257
4271	human3_lib1.final_cell_0656	alpha	186
6844	human3_lib4.final_cell_0548	acinar	299
5257	human3_lib2.final_cell_0756	ductal	297
8536	D28-2_42	acinar	833
9803	D31-7_61	alpha	536
5467	human3_lib3.final_cell_0097	alpha	324
568	human1_lib2.final_cell_0004	acinar	596
3516	human2_lib3.final_cell_0457	delta	145
6918	human3_lib4.final_cell_0622	acinar	179
4111	human3_lib1.final_cell_0496	ductal	369
8000	human4_lib3.final_cell_0224	delta	192
6904	human3_lib4.final_cell_0608	beta	152
9352	D31-2_38	acinar	913
7971	human4_lib3.final_cell_0195	alpha	247
8256	human4_lib3.final_cell_0480	beta	216
7746	human4_lib1.final_cell_0568	t_cell	212
1301	human1_lib3.final_cell_0122	beta	235
8670	D29-1_16	delta	384
2841	human2_lib2.final_cell_0380	alpha	174
7986	human4_lib3.final_cell_0210	alpha	243
6653	human3_lib4.final_cell_0357	beta	188
6657	human3_lib4.final_cell_0361	ductal	368
10294	D30-5_44	ductal	474
7115	human3_lib4.final_cell_0819	PSC	218
6827	human3_lib4.final_cell_0531	delta	94
9116	D29-7_37	beta	513
4198	human3_lib1.final_cell_0583	delta	107
6255	human3_lib3.final_cell_0886	alpha	133
8790	D29-2_63	ductal	784
3671	human3_lib1.final_cell_0055	acinar	319
5415	human3_lib3.final_cell_0044	acinar	500
3144	human2_lib3.final_cell_0085	alpha	205
9796	D31-7_54	alpha	391
9733	D31-6_75	acinar	686
520	human1_lib1.final_cell_0524	beta	124
7022	human3_lib4.final_cell_0726	endothelial	324
5355	human3_lib2.final_cell_0854	alpha	115
8159	human4_lib3.final_cell_0383	alpha	188
1223	human1_lib3.final_cell_0038	beta	269
5457	human3_lib3.final_cell_0087	beta	365
3080	human2_lib3.final_cell_0020	delta	229
7124	human3_lib4.final_cell_0828	PSC	192
10074	D30-2_87	beta	575
7851	human4_lib3.final_cell_0075	ductal	416
9193	D29-8_40	alpha	204
775	human1_lib2.final_cell_0212	ductal	363
10126	D30-3_45	acinar	821
4566	human3_lib2.final_cell_0065	beta	252
10552	D30-8_43	alpha	590
656	human1_lib2.final_cell_0093	endothelial	432
4699	human3_lib2.final_cell_0198	beta	285
951	human1_lib2.final_cell_0388	beta	163
2698	human2_lib2.final_cell_0237	alpha	211
1231	human1_lib3.final_cell_0049	alpha	256
2866	human2_lib2.final_cell_0405	alpha	192
5639	human3_lib3.final_cell_0269	beta	205
3599	human2_lib3.final_cell_0540	ductal	230
4354	human3_lib1.final_cell_0739	endothelial	306
6558	human3_lib4.final_cell_0262	ductal	403
7939	human4_lib3.final_cell_0163	alpha	235
2678	human2_lib2.final_cell_0217	alpha	244
84	human1_lib1.final_cell_0088	beta	235
7074	human3_lib4.final_cell_0778	alpha	166
7937	human4_lib3.final_cell_0161	delta	196
9840	D31-8_10	PP	316
10063	D30-2_76	acinar	863
2693	human2_lib2.final_cell_0232	beta	225
579	human1_lib2.final_cell_0016	acinar	439
6415	human3_lib4.final_cell_0119	PSC	566
7155	human3_lib4.final_cell_0859	PSC	157
7902	human4_lib3.final_cell_0126	beta	311
9025	D29-5_89	ductal	820
2766	human2_lib2.final_cell_0305	alpha	204
584	human1_lib2.final_cell_0021	beta	304
9739	D31-6_82	ductal	826
2003	human2_lib1.final_cell_0098	ductal	521
2164	human2_lib1.final_cell_0259	beta	183
5047	human3_lib2.final_cell_0546	beta	148
9031	D29-6_6	PP	364
10397	D30-6_63	alpha	413
9274	D31-1_48	acinar	854
7574	human4_lib1.final_cell_0396	beta	273
1523	human1_lib3.final_cell_0346	delta	218
4414	human3_lib1.final_cell_0799	alpha	159
334	human1_lib1.final_cell_0338	delta	145
1767	human1_lib3.final_cell_0590	endothelial	340
157	human1_lib1.final_cell_0161	alpha	210
6210	human3_lib3.final_cell_0841	endothelial	290
7355	human4_lib1.final_cell_0177	beta	282
652	human1_lib2.final_cell_0089	beta	223
7446	human4_lib1.final_cell_0268	PP	234
9525	D31-4_42	alpha	498
2984	human2_lib2.final_cell_0523	ductal	362
8681	D29-1_32	alpha	388
5855	human3_lib3.final_cell_0486	alpha	237
5599	human3_lib3.final_cell_0229	acinar	306
6787	human3_lib4.final_cell_0491	ductal	338
1154	human1_lib2.final_cell_0591	alpha	138
194	human1_lib1.final_cell_0198	alpha	242
5782	human3_lib3.final_cell_0413	macrophage	337
1660	human1_lib3.final_cell_0483	delta	137
9838	D31-8_8	beta	441
5846	human3_lib3.final_cell_0477	delta	103
1740	human1_lib3.final_cell_0563	beta	206
6287	human3_lib3.final_cell_0918	beta	100
6938	human3_lib4.final_cell_0642	beta	169
7125	human3_lib4.final_cell_0829	alpha	113
2969	human2_lib2.final_cell_0508	beta	140
7296	human4_lib1.final_cell_0118	ductal	531
7021	human3_lib4.final_cell_0725	alpha	178
8620	D28-4_22	ductal	809
904	human1_lib2.final_cell_0341	alpha	195
8533	D28-2_39	acinar	813
3593	human2_lib3.final_cell_0534	beta	113
2066	human2_lib1.final_cell_0161	alpha	216
8004	human4_lib3.final_cell_0228	beta	287
5788	human3_lib3.final_cell_0419	beta	187
2050	human2_lib1.final_cell_0145	beta	268
152	human1_lib1.final_cell_0156	PSC	547
10078	D30-2_91	alpha	408
2143	human2_lib1.final_cell_0238	PP	267
8626	D28-4_31	beta	825
4275	human3_lib1.final_cell_0660	acinar	198
10281	D30-5_31	ductal	759
6464	human3_lib4.final_cell_0168	alpha	289
6884	human3_lib4.final_cell_0588	beta	145
8812	D29-2_90	alpha	309
9524	D31-4_41	acinar	818
8795	D29-2_68	PP	517
6825	human3_lib4.final_cell_0529	delta	190
955	human1_lib2.final_cell_0392	delta	90
10508	D30-7_89	delta	522
303	human1_lib1.final_cell_0307	beta	151
4758	human3_lib2.final_cell_0257	acinar	221
5857	human3_lib3.final_cell_0488	alpha	223
10060	D30-2_73	alpha	404
10573	D30-8_67	alpha	755
9897	D31-8_77	ductal	988
9921	D30-1_13	ductal	950
5460	human3_lib3.final_cell_0090	acinar	321
9458	D31-3_65	ductal	662
7987	human4_lib3.final_cell_0211	PSC	499
7344	human4_lib1.final_cell_0166	beta	317
6908	human3_lib4.final_cell_0612	alpha	182
6132	human3_lib3.final_cell_0763	alpha	160
1185	human1_lib2.final_cell_0622	endothelial	228
6740	human3_lib4.final_cell_0444	ductal	356
9289	D31-1_64	alpha	575
9783	D31-7_39	beta	535
585	human1_lib2.final_cell_0022	delta	234
3812	human3_lib1.final_cell_0197	acinar	296
5884	human3_lib3.final_cell_0515	beta	116
2330	human2_lib1.final_cell_0425	alpha	173
600	human1_lib2.final_cell_0037	beta	255
2512	human2_lib2.final_cell_0051	alpha	237
571	human1_lib2.final_cell_0007	acinar	413
4473	human3_lib1.final_cell_0858	ductal	174
7406	human4_lib1.final_cell_0228	alpha	308
10381	D30-6_47	beta	500
9490	D31-4_6	alpha	260
6832	human3_lib4.final_cell_0536	PSC	397
3909	human3_lib1.final_cell_0294	beta	221
6145	human3_lib3.final_cell_0776	beta	154
1622	human1_lib3.final_cell_0445	ductal	382
6566	human3_lib4.final_cell_0270	acinar	314
3853	human3_lib1.final_cell_0238	acinar	294
7133	human3_lib4.final_cell_0837	ductal	216
2885	human2_lib2.final_cell_0424	alpha	180
7380	human4_lib1.final_cell_0202	beta	349
10500	D30-7_81	PSC	1015
2134	human2_lib1.final_cell_0229	alpha	181
3953	human3_lib1.final_cell_0338	ductal	384
2438	human2_lib1.final_cell_0533	ductal	274
5929	human3_lib3.final_cell_0560	acinar	261
5682	human3_lib3.final_cell_0313	beta	196
5157	human3_lib2.final_cell_0656	alpha	196
5837	human3_lib3.final_cell_0468	alpha	212
9643	D31-5_74	beta	393
6831	human3_lib4.final_cell_0535	alpha	218
5345	human3_lib2.final_cell_0844	alpha	102
7969	human4_lib3.final_cell_0193	alpha	204
8565	D28-3_8	alpha	399
8203	human4_lib3.final_cell_0427	endothelial	343
6834	human3_lib4.final_cell_0538	ductal	342
8046	human4_lib3.final_cell_0270	alpha	202
7452	human4_lib1.final_cell_0274	beta	321
301	human1_lib1.final_cell_0305	beta	166
5231	human3_lib2.final_cell_0730	alpha	180
2657	human2_lib2.final_cell_0196	beta	252
1811	human1_lib3.final_cell_0634	beta	152
3834	human3_lib1.final_cell_0219	beta	240
789	human1_lib2.final_cell_0226	PP	268
552	human1_lib1.final_cell_0556	beta	102
858	human1_lib2.final_cell_0295	beta	215
4935	human3_lib2.final_cell_0434	beta	177
5515	human3_lib3.final_cell_0145	acinar	234
7468	human4_lib1.final_cell_0290	beta	293
9111	D29-7_32	alpha	353
1911	human1_lib3.final_cell_0734	beta	97
5617	human3_lib3.final_cell_0247	beta	211
192	human1_lib1.final_cell_0196	beta	199
1944	human2_lib1.final_cell_0036	delta	203
4614	human3_lib2.final_cell_0113	beta	297
727	human1_lib2.final_cell_0164	beta	233
```
<!-- END_RESTORE_FILE -->


</details>

## file-65aa18e55ff1

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/inference_input_cells.tsv` · SHA256: `3e2165db6ac90ff77b64dec618d9855cb2ee5806ce22f60bddeeb8f9ca18ac84`

<details>
<summary>inference_input_cells.tsv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/inference_input_cells.tsv", "sha256": "3e2165db6ac90ff77b64dec618d9855cb2ee5806ce22f60bddeeb8f9ca18ac84", "encoding": "utf-8", "added_final_newline": false} -->
```tsv
original_row_index	cell_id	Celltype	tokens_before_sampling_including_cls
18	10th_C43_S50	alpha	763
19	10th_C45_S66	alpha	883
22	10th_C51_S80	beta	684
23	10th_C52_S110	alpha	867
24	10th_C53_S118	alpha	711
26	10th_C56_S74	PSC	1078
30	10th_C63_S82	acinar	961
34	10th_C76_S128	delta	839
48	11th-C21_S4	beta	796
49	11th-C22_S65	beta	866
50	11th-C24_S49	ductal	1092
59	11th-C41_S60	beta	1031
64	11th-C51_S39	beta	965
65	11th-C53_S78	beta	867
67	11th-C56_S32	PSC	1034
76	11th-C68_S34	delta	972
79	11th-C93_S30	alpha	737
91	12th-C28_S142	alpha	883
96	12th-C36_S158	PSC	931
103	12th-C46_S145	PSC	1109
114	12th-C59_S169	beta	943
116	12th-C63_S219	alpha	791
117	12th-C64_S163	beta	921
124	12th-C7_S196	beta	866
129	12th-C8_S189	beta	597
132	12th-C94_S180	PSC	1084
136	13th_C13_S149	alpha	928
143	13th_C21_S135	beta	833
144	13th_C23_S187	alpha	1058
146	13th_C29_S188	alpha	610
151	13th_C39_S153	beta	941
160	13th_C55_S156	alpha	755
161	13th_C57_S171	beta	799
164	13th_C63_S172	beta	788
174	13th_C77_S209	alpha	625
177	13th_C81_S160	beta	955
186	1st-61_S27	beta	947
196	1st-C39_S23	beta	782
197	1st-C42_S71	PSC	1243
200	1st-C53_S74	alpha	745
207	1st-C68_S36	alpha	1044
208	1st-C69_S44	delta	926
210	1st-C73_S45	alpha	875
217	1st-C8_S10	alpha	681
221	2nd-C15_S25	beta	950
224	2nd-C19_S39	beta	756
227	2nd-C25_S49	delta	894
233	2nd-C31_S61	alpha	949
234	2nd-C32_S62	beta	912
243	2nd-C51_S6	beta	840
248	2nd-C5_S8	beta	947
250	2nd-C67_S40	alpha	939
264	2nd-C88_S84	beta	922
272	3rd-C1_S39	delta	890
277	3rd-C26_S28	alpha	896
280	3rd-C33_S50	beta	1060
287	3rd-C4_S25	beta	979
289	3rd-C52_S86	alpha	833
292	3rd-C58_S3	beta	886
295	3rd-C61_S64	alpha	884
299	3rd-C68_S13	delta	883
303	3rd-C76_S22	alpha	874
304	3rd-C79_S67	alpha	1042
305	3rd-C7_S40	alpha	832
307	3rd-C84_S75	beta	1103
321	4th-C18_S31	alpha	905
325	4th-C26_S50	beta	1138
327	4th-C28_S55	beta	956
329	4th-C32_S62	delta	959
331	4th-C35_S68	alpha	754
339	4th-C50_S5	alpha	675
342	4th-C57_S18	alpha	745
346	4th-C63_S30	alpha	714
352	4th-C73_S54	alpha	865
354	4th-C78_S58	alpha	1138
359	4th-C82_S72	alpha	1022
365	4th-C89_S83	beta	950
370	4th-C95_S95	acinar	1015
372	4th-C9_S13	alpha	745
377	5th-C16_S98	beta	847
378	5th-C18_S96	beta	787
381	5th-C25_S108	beta	647
382	5th-C30_S105	ductal	1000
383	5th-C34_S108	alpha	933
386	5th-C41_S113	ductal	893
389	5th-C45_S123	acinar	1124
393	5th-C54_S89	acinar	1009
399	5th-C66_S101	alpha	861
403	5th-C70_S103	alpha	785
405	5th-C74_S110	alpha	899
410	5th-C87_S119	alpha	811
414	6th-C12_S18	beta	1075
415	6th-C15_S22	beta	682
419	6th-C20_S34	beta	842
422	6th-C23_S38	beta	1050
426	6th-C29_S50	beta	921
427	6th-C2_S2	beta	976
429	6th-C34_S61	alpha	740
433	6th-C51_S6	beta	875
438	6th-C5_S8	alpha	898
441	6th-C64_S30	beta	750
442	6th-C65_S31	alpha	933
445	6th-C71_S41	beta	1028
452	6th-C83_S65	alpha	854
461	7th-C10_S19	beta	1045
463	7th-C12_S17	beta	1027
465	7th-C15_S23	delta	860
466	7th-C19_S36	beta	665
471	7th-C28_S48	beta	966
473	7th-C31_S53	beta	959
478	7th-C44_S76	acinar	768
484	7th-C52_S9	beta	844
485	7th-C53_S10	beta	977
487	7th-C57_S16	beta	814
490	7th-C60_S22	beta	763
491	7th-C62_S27	beta	979
494	7th-C66_S33	beta	885
496	7th-C69_S39	alpha	796
501	7th-C79_S58	alpha	1048
502	7th-C80_S57	beta	933
505	7th-C85_S69	alpha	691
507	7th-C8_S13	beta	891
508	7th-C90_S73	beta	1082
511	8th-C10_S100	beta	1049
513	8th-C18_S110	beta	778
514	8th-C20_S116	alpha	805
516	8th-C23_S121	alpha	621
517	8th-C24_S120	beta	895
521	8th-C34_S141	alpha	841
528	8th-C58_S101	alpha	985
530	8th-C61_S107	beta	905
535	8th-C70_S123	beta	713
536	8th-C73_S130	ductal	1115
538	8th-C75_S128	beta	706
541	8th-C84_S143	delta	663
544	8th-C90_S152	alpha	729
546	8th-C95_S161	alpha	800
547	9th-C10_S18	beta	960
548	9th-C11_S17	beta	848
551	9th-C15_S22	alpha	958
553	9th-C17_S29	alpha	839
557	9th-C22_S41	PSC	829
558	9th-C25_S45	beta	697
564	9th-C3_S1	beta	819
565	9th-C41_S70	beta	942
576	9th-C65_S32	beta	898
579	9th-C6_S7	beta	763
584	9th-C80_S56	alpha	927
586	9th-C83_S61	alpha	968
587	9th-C84_S60	ductal	959
591	9th-C88_S73	beta	673
593	9th-C8_S14	beta	845
601	AZ-A12	delta	266
609	AZ-B12	alpha	398
611	AZ-B3	alpha	573
616	AZ-C10	beta	371
620	AZ-C4	beta	365
622	AZ-C8	alpha	429
623	AZ-C9	ductal	877
625	AZ-D2	beta	345
632	AZ-D9	alpha	314
634	AZ-E2	alpha	417
635	AZ-E3	MHC class II	419
636	AZ-E4	alpha	443
641	AZ-F1	beta	459
644	AZ-F5	PSC	690
647	AZ-F9	ductal	569
650	AZ-G2	beta	242
653	AZ-G5	PP	346
656	AZ-H11	beta	291
657	AZ-H12	PP	252
661	AZ-H5	PP	325
663	AZ-H9	alpha	258
669	HP1502401-A18	delta	503
671	HP1502401-A22	acinar	636
685	HP1502401-B21	beta	439
686	HP1502401-B23	beta	412
696	HP1502401-C18	beta	311
700	HP1502401-C5	delta	433
704	HP1502401-D10	acinar	809
710	HP1502401-D2	acinar	912
714	HP1502401-D3	beta	459
719	HP1502401-D8	alpha	359
721	HP1502401-E11	alpha	411
727	HP1502401-E20	acinar	897
732	HP1502401-E5	PP	422
738	HP1502401-F12	alpha	434
747	HP1502401-F22	beta	277
757	HP1502401-G20	delta	601
783	HP1502401-H4	alpha	374
785	HP1502401-H6	delta	378
787	HP1502401-H8	alpha	385
792	HP1502401-I16	alpha	394
794	HP1502401-I2	ductal	596
800	HP1502401-I6	beta	517
803	HP1502401-I9	delta	381
813	HP1502401-J19	ductal	734
815	HP1502401-J20	alpha	567
819	HP1502401-J24	ductal	552
823	HP1502401-J8	alpha	371
825	HP1502401-K10	acinar	797
829	HP1502401-K15	alpha	302
833	HP1502401-K21	alpha	569
838	HP1502401-K5	alpha	384
861	HP1502401-M1	alpha	348
863	HP1502401-M11	alpha	498
867	HP1502401-M16	ductal	544
882	HP1502401-N10	ductal	753
893	HP1502401-N3	alpha	431
895	HP1502401-N7	delta	459
901	HP1502401-O16	beta	379
903	HP1502401-O18	delta	492
907	HP1502401-O22	alpha	367
910	HP1502401-O3	alpha	341
912	HP1502401-O6	delta	363
914	HP1502401-O9	mast	263
922	HP1504101T2D-A20	PP	438
923	HP1504101T2D-A23	alpha	517
927	HP1504101T2D-A6	alpha	424
929	HP1504101T2D-B10	PP	449
934	HP1504101T2D-B17	PP	521
937	HP1504101T2D-B22	PP	504
944	HP1504101T2D-C1	endothelial	685
946	HP1504101T2D-C11	acinar	914
948	HP1504101T2D-C14	alpha	423
955	HP1504101T2D-C4	alpha	381
957	HP1504101T2D-D1	alpha	465
965	HP1504101T2D-D2	alpha	474
971	HP1504101T2D-D7	alpha	467
975	HP1504101T2D-E11	PP	461
989	HP1504101T2D-E7	PP	403
991	HP1504101T2D-E9	PP	516
993	HP1504101T2D-F10	alpha	369
995	HP1504101T2D-F12	alpha	471
997	HP1504101T2D-F14	PP	411
1000	HP1504101T2D-F17	delta	568
1001	HP1504101T2D-F18	PSC	838
1014	HP1504101T2D-G19	alpha	513
1019	HP1504101T2D-G24	alpha	529
1020	HP1504101T2D-G3	PP	420
1022	HP1504101T2D-G8	PP	558
1027	HP1504101T2D-H17	alpha	535
1028	HP1504101T2D-H18	ductal	964
1032	HP1504101T2D-H22	alpha	453
1035	HP1504101T2D-H4	alpha	346
1036	HP1504101T2D-H5	delta	507
1044	HP1504101T2D-I17	alpha	435
1050	HP1504101T2D-I24	alpha	579
1052	HP1504101T2D-I6	acinar	737
1053	HP1504101T2D-I7	PP	529
1065	HP1504101T2D-J24	alpha	552
1067	HP1504101T2D-J5	PP	575
1068	HP1504101T2D-J7	alpha	456
1070	HP1504101T2D-K10	acinar	850
1071	HP1504101T2D-K12	alpha	636
1076	HP1504101T2D-K20	alpha	535
1078	HP1504101T2D-K23	alpha	554
1083	HP1504101T2D-L10	alpha	439
1087	HP1504101T2D-L15	PP	453
1100	HP1504101T2D-M10	alpha	493
1101	HP1504101T2D-M15	alpha	392
1112	HP1504101T2D-M7	alpha	485
1113	HP1504101T2D-M8	epsilon	426
1117	HP1504101T2D-N12	alpha	463
1122	HP1504101T2D-N19	alpha	512
1127	HP1504101T2D-N24	alpha	457
1132	HP1504101T2D-N9	alpha	383
1134	HP1504101T2D-O11	PP	538
1144	HP1504101T2D-O4	PP	491
1145	HP1504101T2D-O6	alpha	548
1149	HP1504101T2D-P14	alpha	1069
1156	HP1504101T2D-P3	PSC	730
1158	HP1504101T2D-P7	alpha	486
1159	HP1504101T2D-P9	PP	457
1160	HP1504901-A10	delta	360
1161	HP1504901-A14	alpha	257
1164	HP1504901-A2	alpha	346
1166	HP1504901-A22	PSC	414
1169	HP1504901-A5	PP	364
1172	HP1504901-B10	beta	310
1185	HP1504901-B6	PP	355
1190	HP1504901-C13	epsilon	357
1192	HP1504901-C16	alpha	301
1199	HP1504901-D15	alpha	265
1201	HP1504901-D17	alpha	390
1202	HP1504901-D18	epsilon	448
1204	HP1504901-D21	alpha	347
1209	HP1504901-D6	delta	257
1213	HP1504901-E16	beta	283
1217	HP1504901-E23	PP	364
1222	HP1504901-F10	alpha	313
1234	HP1504901-F23	acinar	568
1235	HP1504901-F24	alpha	394
1238	HP1504901-F5	alpha	326
1241	HP1504901-G10	endothelial	517
1247	HP1504901-G19	alpha	314
1249	HP1504901-G20	PP	280
1251	HP1504901-G24	alpha	211
1252	HP1504901-G3	alpha	325
1260	HP1504901-H12	alpha	333
1266	HP1504901-H2	alpha	185
1268	HP1504901-H4	alpha	375
1270	HP1504901-H6	alpha	293
1279	HP1504901-I22	PP	412
1283	HP1504901-I8	alpha	488
1297	HP1504901-J9	PP	345
1300	HP1504901-K14	PP	327
1305	HP1504901-K5	alpha	330
1316	HP1504901-L3	PP	367
1319	HP1504901-M11	PSC	423
1325	HP1504901-M3	alpha	372
1327	HP1504901-N1	alpha	331
1329	HP1504901-N13	beta	313
1338	HP1504901-N23	PP	317
1342	HP1504901-N9	PP	368
1348	HP1504901-O17	alpha	353
1366	HP1504901-P7	ductal	469
1373	HP1506401-A21	alpha	734
1374	HP1506401-A24	beta	663
1383	HP1506401-B15	PSC	1018
1390	HP1506401-B3	PP	808
1395	HP1506401-C15	beta	534
1409	HP1506401-D15	ductal	1176
1413	HP1506401-D3	acinar	966
1414	HP1506401-D7	ductal	995
1417	HP1506401-E11	alpha	522
1436	HP1506401-F2	acinar	1078
1440	HP1506401-F3	acinar	844
1441	HP1506401-F5	acinar	1143
1442	HP1506401-F8	beta	999
1443	HP1506401-F9	beta	764
1454	HP1506401-G21	ductal	996
1455	HP1506401-G23	ductal	1101
1458	HP1506401-G4	acinar	1061
1460	HP1506401-G6	beta	660
1461	HP1506401-G8	PP	851
1463	HP1506401-H1	acinar	1068
1464	HP1506401-H13	alpha	722
1466	HP1506401-H15	acinar	1047
1467	HP1506401-H16	acinar	1040
1472	HP1506401-H23	ductal	1055
1473	HP1506401-H3	acinar	1023
1475	HP1506401-H6	ductal	987
1483	HP1506401-I14	PP	656
1493	HP1506401-I8	alpha	652
1500	HP1506401-J2	acinar	1111
1518	HP1506401-K17	ductal	1076
1519	HP1506401-K19	ductal	1024
1520	HP1506401-K2	acinar	1155
1523	HP1506401-K23	PP	673
1524	HP1506401-K4	ductal	839
1527	HP1506401-L10	ductal	965
1529	HP1506401-L13	ductal	1009
1531	HP1506401-L18	acinar	1076
1532	HP1506401-L19	acinar	608
1536	HP1506401-L23	alpha	1099
1542	HP1506401-M1	acinar	1044
1543	HP1506401-M11	ductal	1031
1545	HP1506401-M16	ductal	1038
1554	HP1506401-N10	ductal	1111
1557	HP1506401-N16	ductal	930
1565	HP1506401-N5	acinar	909
1566	HP1506401-N9	acinar	957
1567	HP1506401-O10	beta	601
1568	HP1506401-O11	beta	653
1572	HP1506401-O2	ductal	1034
1574	HP1506401-O23	alpha	685
1577	HP1506401-O4	ductal	968
1583	HP1506401-P12	beta	671
1584	HP1506401-P14	beta	1262
1588	HP1506401-P22	ductal	1040
1594	HP1507101-A14	alpha	463
1597	HP1507101-A4	alpha	679
1601	HP1507101-B12	alpha	510
1616	HP1507101-C14	alpha	843
1620	HP1507101-C5	alpha	555
1629	HP1507101-D18	alpha	728
1632	HP1507101-D23	beta	495
1634	HP1507101-D5	alpha	539
1653	HP1507101-F13	alpha	844
1660	HP1507101-F20	alpha	574
1661	HP1507101-F22	alpha	642
1664	HP1507101-F4	beta	599
1670	HP1507101-G13	alpha	363
1673	HP1507101-G16	alpha	575
1680	HP1507101-H1	alpha	570
1683	HP1507101-H14	alpha	653
1685	HP1507101-H16	ductal	1081
1686	HP1507101-H17	beta	619
1690	HP1507101-H4	beta	588
1691	HP1507101-H5	alpha	676
1693	HP1507101-H7	alpha	486
1697	HP1507101-I13	alpha	837
1700	HP1507101-I18	beta	751
1701	HP1507101-I19	alpha	371
1702	HP1507101-I2	alpha	665
1703	HP1507101-I20	alpha	670
1705	HP1507101-I4	alpha	558
1708	HP1507101-I8	alpha	645
1710	HP1507101-J11	alpha	650
1714	HP1507101-J17	alpha	645
1718	HP1507101-J20	beta	836
1720	HP1507101-J22	alpha	590
1725	HP1507101-J7	acinar	978
1730	HP1507101-K23	beta	698
1734	HP1507101-L1	PSC	1066
1739	HP1507101-L18	alpha	766
1740	HP1507101-L19	alpha	493
1753	HP1507101-M22	PP	436
1771	HP1507101-O14	beta	283
1772	HP1507101-O16	alpha	741
1776	HP1507101-O23	delta	472
1777	HP1507101-O3	beta	518
1779	HP1507101-O8	beta	539
1785	HP1507101-P23	alpha	376
1789	HP1507101-P8	alpha	373
1790	HP1508501T2D-A10	beta	546
1803	HP1508501T2D-A8	PSC	1143
1808	HP1508501T2D-B15	alpha	664
1814	HP1508501T2D-B23	alpha	470
1817	HP1508501T2D-B5	alpha	356
1821	HP1508501T2D-C11	ductal	1138
1825	HP1508501T2D-C15	ductal	982
1829	HP1508501T2D-C19	ductal	834
1830	HP1508501T2D-C2	ductal	850
1832	HP1508501T2D-C23	alpha	647
1834	HP1508501T2D-C8	delta	685
1835	HP1508501T2D-C9	MHC class II	980
1836	HP1508501T2D-D1	alpha	460
1838	HP1508501T2D-D11	alpha	499
1840	HP1508501T2D-D14	alpha	651
1841	HP1508501T2D-D15	alpha	876
1842	HP1508501T2D-D16	alpha	595
1849	HP1508501T2D-D3	alpha	635
1850	HP1508501T2D-D5	alpha	578
1854	HP1508501T2D-E1	PSC	1302
1857	HP1508501T2D-E12	alpha	657
1859	HP1508501T2D-E14	alpha	476
1861	HP1508501T2D-E16	ductal	854
1862	HP1508501T2D-E17	ductal	986
1864	HP1508501T2D-E19	alpha	702
1865	HP1508501T2D-E2	alpha	426
1868	HP1508501T2D-E22	alpha	738
1874	HP1508501T2D-E9	ductal	1218
1880	HP1508501T2D-F16	alpha	736
1885	HP1508501T2D-F21	acinar	909
1890	HP1508501T2D-F7	beta	655
1892	HP1508501T2D-F9	alpha	379
1893	HP1508501T2D-G1	ductal	977
1895	HP1508501T2D-G11	alpha	630
1896	HP1508501T2D-G12	acinar	864
1901	HP1508501T2D-G18	acinar	1037
1902	HP1508501T2D-G19	acinar	968
1904	HP1508501T2D-G20	alpha	538
1910	HP1508501T2D-G5	beta	483
1912	HP1508501T2D-H10	alpha	605
1914	HP1508501T2D-H12	mast	497
1919	HP1508501T2D-H17	ductal	924
1923	HP1508501T2D-H23	alpha	474
1926	HP1508501T2D-H5	PSC	933
1937	HP1508501T2D-I18	delta	772
1940	HP1508501T2D-I20	alpha	517
1942	HP1508501T2D-I22	alpha	634
1943	HP1508501T2D-I23	delta	679
1953	HP1508501T2D-J15	ductal	889
1955	HP1508501T2D-J17	alpha	873
1967	HP1508501T2D-J9	alpha	574
1968	HP1508501T2D-K1	alpha	657
1973	HP1508501T2D-K17	alpha	689
1978	HP1508501T2D-K22	ductal	1034
1987	HP1508501T2D-L13	alpha	632
1988	HP1508501T2D-L14	alpha	569
1995	HP1508501T2D-L21	ductal	960
2000	HP1508501T2D-L9	ductal	915
2004	HP1508501T2D-M12	alpha	644
2020	HP1508501T2D-N1	ductal	1099
2022	HP1508501T2D-N11	alpha	522
2024	HP1508501T2D-N15	ductal	871
2027	HP1508501T2D-N19	ductal	757
2029	HP1508501T2D-N20	alpha	410
2037	HP1508501T2D-N8	ductal	1053
2039	HP1508501T2D-O1	alpha	673
2040	HP1508501T2D-O11	acinar	994
2043	HP1508501T2D-O14	alpha	392
2049	HP1508501T2D-P11	alpha	696
2053	HP1508501T2D-P4	PP	713
2057	HP1508501T2D-P9	alpha	718
2061	HP1509101-B10	alpha	720
2063	HP1509101-B16	acinar	1017
2065	HP1509101-B19	ductal	955
2069	HP1509101-C22	alpha	548
2071	HP1509101-D17	ductal	917
2073	HP1509101-D19	alpha	584
2080	HP1509101-E3	ductal	578
2088	HP1509101-F7	ductal	787
2095	HP1509101-H20	ductal	890
2108	HP1509101-J16	delta	396
2110	HP1509101-J20	alpha	657
2112	HP1509101-J23	ductal	735
2113	HP1509101-J4	alpha	632
2116	HP1509101-K10	ductal	1170
2125	HP1509101-L17	alpha	469
2135	HP1509101-N23	beta	429
2141	HP1509101-O18	beta	400
2148	HP1509101-P20	alpha	543
2149	HP1509101-P23	PSC	994
2162	HP1525301T2D-A3	alpha	598
2163	HP1525301T2D-A5	alpha	615
2164	HP1525301T2D-A6	alpha	715
2167	HP1525301T2D-B1	alpha	584
2169	HP1525301T2D-B11	alpha	571
2176	HP1525301T2D-B24	ductal	831
2177	HP1525301T2D-B3	ductal	933
2181	HP1525301T2D-B7	alpha	532
2186	HP1525301T2D-C13	ductal	612
2192	HP1525301T2D-C2	alpha	592
2193	HP1525301T2D-C21	ductal	1049
2197	HP1525301T2D-C3	ductal	922
2200	HP1525301T2D-C6	ductal	1086
2205	HP1525301T2D-D10	acinar	950
2209	HP1525301T2D-D15	ductal	979
2210	HP1525301T2D-D19	delta	642
2211	HP1525301T2D-D2	ductal	898
2214	HP1525301T2D-D23	beta	570
2216	HP1525301T2D-D4	ductal	835
2218	HP1525301T2D-D8	ductal	1025
2221	HP1525301T2D-E11	ductal	669
2226	HP1525301T2D-E17	ductal	1020
2230	HP1525301T2D-E23	alpha	496
2231	HP1525301T2D-E24	PSC	1065
2232	HP1525301T2D-E3	PP	732
2233	HP1525301T2D-E5	PSC	982
2238	HP1525301T2D-F15	beta	650
2239	HP1525301T2D-F16	ductal	675
2243	HP1525301T2D-F23	beta	615
2244	HP1525301T2D-F24	PSC	1082
2245	HP1525301T2D-F3	acinar	949
2246	HP1525301T2D-F4	alpha	555
2247	HP1525301T2D-F5	acinar	1091
2249	HP1525301T2D-F8	ductal	893
2259	HP1525301T2D-G20	ductal	984
2260	HP1525301T2D-G21	ductal	848
2262	HP1525301T2D-G23	PP	601
2263	HP1525301T2D-G3	ductal	997
2284	HP1525301T2D-H8	acinar	605
2291	HP1525301T2D-I16	ductal	858
2294	HP1525301T2D-I19	ductal	1079
2298	HP1525301T2D-I23	ductal	982
2310	HP1525301T2D-J15	ductal	1136
2311	HP1525301T2D-J16	acinar	997
2313	HP1525301T2D-J18	PP	635
2318	HP1525301T2D-J23	acinar	887
2319	HP1525301T2D-J4	alpha	677
2321	HP1525301T2D-J7	ductal	1045
2330	HP1525301T2D-K16	ductal	1100
2342	HP1525301T2D-K8	acinar	971
2346	HP1525301T2D-L15	alpha	529
2349	HP1525301T2D-L2	ductal	916
2350	HP1525301T2D-L20	ductal	1174
2353	HP1525301T2D-L24	alpha	626
2355	HP1525301T2D-L5	acinar	961
2361	HP1525301T2D-M12	ductal	931
2364	HP1525301T2D-M16	alpha	601
2368	HP1525301T2D-M22	ductal	302
2369	HP1525301T2D-M5	acinar	1045
2373	HP1525301T2D-M9	ductal	1034
2375	HP1525301T2D-N10	acinar	729
2377	HP1525301T2D-N12	ductal	976
2389	HP1525301T2D-N6	ductal	1100
2394	HP1525301T2D-O11	PP	547
2396	HP1525301T2D-O14	ductal	1077
2397	HP1525301T2D-O16	ductal	824
2402	HP1525301T2D-O22	ductal	993
2415	HP1525301T2D-P16	PSC	1083
2420	HP1525301T2D-P20	ductal	863
2422	HP1525301T2D-P22	alpha	648
2425	HP1525301T2D-P4	ductal	1018
2428	HP1525301T2D-P8	alpha	632
2433	HP1526901T2D-A12	alpha	890
2434	HP1526901T2D-A14	beta	630
2436	HP1526901T2D-A16	delta	621
2441	HP1526901T2D-A21	ductal	983
2442	HP1526901T2D-A22	delta	888
2446	HP1526901T2D-A5	beta	684
2450	HP1526901T2D-A9	alpha	638
2452	HP1526901T2D-B10	beta	812
2453	HP1526901T2D-B11	alpha	656
2458	HP1526901T2D-B18	delta	673
2460	HP1526901T2D-B23	ductal	819
2461	HP1526901T2D-B24	ductal	1008
2462	HP1526901T2D-B3	ductal	1079
2464	HP1526901T2D-B5	ductal	940
2467	HP1526901T2D-B9	beta	627
2471	HP1526901T2D-C13	PP	598
2475	HP1526901T2D-C17	beta	579
2476	HP1526901T2D-C19	PP	689
2484	HP1526901T2D-C6	alpha	626
2485	HP1526901T2D-C8	acinar	933
2492	HP1526901T2D-D18	alpha	634
2513	HP1526901T2D-E20	delta	625
2515	HP1526901T2D-E22	ductal	1120
2517	HP1526901T2D-E3	delta	713
2520	HP1526901T2D-E6	alpha	601
2525	HP1526901T2D-F12	PP	761
2527	HP1526901T2D-F16	ductal	1021
2529	HP1526901T2D-F18	beta	591
2534	HP1526901T2D-F22	ductal	870
2537	HP1526901T2D-F3	alpha	693
2539	HP1526901T2D-F7	PP	750
2542	HP1526901T2D-G1	PP	689
2543	HP1526901T2D-G10	alpha	711
2548	HP1526901T2D-G17	beta	627
2550	HP1526901T2D-G19	PP	592
2559	HP1526901T2D-G9	PP	697
2563	HP1526901T2D-H17	alpha	678
2564	HP1526901T2D-H18	delta	616
2570	HP1526901T2D-H23	PP	679
2581	HP1526901T2D-I12	acinar	1110
2588	HP1526901T2D-I19	delta	650
2592	HP1526901T2D-I24	PP	684
2594	HP1526901T2D-I4	alpha	598
2597	HP1526901T2D-I8	alpha	555
2599	HP1526901T2D-J1	delta	591
2600	HP1526901T2D-J11	ductal	1107
2601	HP1526901T2D-J12	ductal	1024
2609	HP1526901T2D-J21	beta	610
2615	HP1526901T2D-J6	beta	709
2616	HP1526901T2D-J8	alpha	547
2617	HP1526901T2D-K1	ductal	1131
2618	HP1526901T2D-K10	alpha	655
2623	HP1526901T2D-K17	acinar	1096
2624	HP1526901T2D-K18	alpha	650
2627	HP1526901T2D-K20	beta	601
2630	HP1526901T2D-K23	beta	564
2633	HP1526901T2D-K4	ductal	1033
2638	HP1526901T2D-L12	PP	619
2642	HP1526901T2D-L16	alpha	603
2647	HP1526901T2D-L23	alpha	672
2648	HP1526901T2D-L24	delta	599
2650	HP1526901T2D-L5	beta	754
2654	HP1526901T2D-M13	epsilon	698
2661	HP1526901T2D-M20	ductal	1096
2663	HP1526901T2D-M22	PP	649
2664	HP1526901T2D-M23	alpha	644
2665	HP1526901T2D-M3	PP	665
2668	HP1526901T2D-M8	ductal	1100
2669	HP1526901T2D-N1	beta	640
2670	HP1526901T2D-N10	alpha	664
2673	HP1526901T2D-N13	beta	637
2678	HP1526901T2D-N19	PP	631
2685	HP1526901T2D-N6	alpha	630
2687	HP1526901T2D-N8	alpha	504
2689	HP1526901T2D-O1	beta	590
2694	HP1526901T2D-O15	beta	680
2695	HP1526901T2D-O17	beta	584
2697	HP1526901T2D-O2	PSC	1040
2698	HP1526901T2D-O21	alpha	691
2707	HP1526901T2D-O8	beta	636
2718	HP1526901T2D-P2	alpha	513
2719	HP1526901T2D-P20	alpha	578
2722	HP1526901T2D-P23	PP	585
2723	HP1526901T2D-P4	beta	553
2728	Sample_3	beta	435
2736	Sample_11	beta	565
2742	Sample_17	beta	325
2748	Sample_23	beta	313
2749	Sample_24	beta	348
2753	Sample_28	beta	385
2754	Sample_29	beta	443
2756	Sample_32	beta	612
2759	Sample_35	beta	525
2760	Sample_36	beta	413
2763	Sample_39	beta	631
2766	Sample_42	beta	493
2768	Sample_45	beta	580
2769	Sample_46	beta	548
2778	Sample_55	beta	544
2782	Sample_59	beta	417
2785	Sample_62	beta	391
2786	Sample_63	beta	356
2789	Sample_66	beta	384
2793	Sample_71	beta	411
2795	Sample_74	beta	527
2814	Sample_93	beta	373
2820	Sample_100	beta	280
2823	Sample_103	beta	378
2825	Sample_105	beta	436
2826	Sample_106	beta	459
2833	Sample_113	beta	396
2836	Sample_117	beta	365
2845	Sample_128	beta	488
2851	Sample_134	beta	486
2854	Sample_137	beta	500
2858	Sample_141	beta	527
2860	Sample_143	beta	388
2865	Sample_150	beta	362
2870	Sample_155	beta	386
2872	Sample_157	beta	408
2873	Sample_158	beta	449
2878	Sample_163	beta	355
2882	Sample_167	beta	389
2888	Sample_175	beta	396
2889	Sample_176	beta	530
2898	Sample_185	beta	430
2902	Sample_189	beta	309
2904	Sample_191	beta	359
2908	Sample_195	beta	399
2916	Sample_203	beta	492
2917	Sample_205	beta	577
2921	Sample_209	alpha	425
2926	Sample_215	alpha	427
2931	Sample_220	alpha	479
2947	Sample_237	alpha	431
2948	Sample_238	alpha	464
2949	Sample_239	alpha	602
2955	Sample_245	alpha	464
2968	Sample_259	alpha	610
2969	Sample_260	alpha	484
2974	Sample_265	alpha	257
2975	Sample_266	alpha	468
2986	Sample_279	alpha	402
2993	Sample_286	alpha	395
2996	Sample_291	alpha	525
3003	Sample_298	alpha	413
3004	Sample_299	alpha	499
3014	Sample_309	alpha	315
3015	Sample_310	alpha	449
3019	Sample_314	alpha	449
3024	Sample_319	alpha	411
3026	Sample_321	alpha	348
3047	Sample_344	alpha	572
3049	Sample_346	alpha	430
3052	Sample_349	alpha	453
3057	Sample_354	alpha	428
3058	Sample_355	alpha	376
3059	Sample_356	alpha	410
3060	Sample_357	alpha	423
3068	Sample_365	alpha	394
3082	Sample_380	alpha	424
3085	Sample_383	alpha	372
3091	Sample_389	alpha	421
3093	Sample_391	alpha	403
3099	Sample_397	alpha	407
3102	Sample_400	alpha	462
3103	Sample_401	alpha	556
3107	Sample_405	alpha	476
3108	Sample_406	alpha	413
3114	Sample_412	alpha	403
3123	Sample_423	alpha	504
3126	Sample_426	alpha	664
3127	Sample_427	alpha	514
3129	Sample_429	alpha	684
3137	Sample_439	alpha	323
3143	Sample_445	alpha	340
3144	Sample_446	alpha	350
3149	Sample_451	alpha	306
3150	Sample_452	alpha	208
3151	Sample_453	alpha	429
3156	Sample_458	alpha	501
3158	Sample_460	alpha	465
3160	Sample_462	alpha	422
3167	Sample_469	alpha	316
3168	Sample_470	alpha	473
3175	Sample_477	alpha	410
3180	Sample_482	alpha	445
3184	Sample_486	alpha	319
3186	Sample_489	alpha	476
3188	Sample_492	alpha	431
3189	Sample_493	alpha	418
3192	Sample_496	alpha	498
3200	Sample_504	alpha	393
3212	Sample_517	alpha	359
3215	Sample_520	alpha	347
3217	Sample_522	alpha	463
3219	Sample_524	alpha	450
3221	Sample_526	alpha	424
3226	Sample_533	alpha	413
3228	Sample_535	alpha	315
3234	Sample_541	alpha	552
3242	Sample_550	alpha	406
3244	Sample_552	alpha	457
3262	Sample_572	alpha	497
3266	Sample_577	alpha	378
3275	Sample_586	delta	589
3277	Sample_588	delta	587
3278	Sample_589	delta	631
3282	Sample_593	delta	483
3286	Sample_597	delta	472
3287	Sample_598	delta	448
3292	Sample_603	delta	431
3298	Sample_609	delta	370
3301	Sample_612	delta	347
3304	Sample_615	PP	608
3308	Sample_619	PP	328
3311	Sample_622	PP	473
3313	Sample_624	PP	480
3320	Sample_631	PP	675
3330	Sample_644	PP	421
3334	Sample_651	PP	558
3337	Sample_654	beta	464
3338	Sample_655	beta	501
3344	Sample_661	beta	425
3347	Sample_664	beta	571
3348	Sample_666	beta	495
3363	Sample_683	beta	459
3365	Sample_685	beta	300
3372	Sample_692	beta	403
3375	Sample_695	beta	415
3376	Sample_696	beta	429
3379	Sample_699	beta	327
3384	Sample_704	beta	391
3388	Sample_708	beta	423
3390	Sample_711	beta	331
3391	Sample_712	beta	396
3392	Sample_713	beta	372
3394	Sample_715	beta	354
3400	Sample_722	beta	548
3404	Sample_726	beta	497
3412	Sample_736	beta	379
3413	Sample_737	beta	304
3423	Sample_748	beta	355
3424	Sample_749	beta	484
3429	Sample_754	beta	355
3439	Sample_764	beta	402
3440	Sample_765	beta	436
3442	Sample_767	beta	425
3443	Sample_769	beta	332
3447	Sample_773	beta	422
3451	Sample_777	beta	345
3454	Sample_780	beta	253
3457	Sample_783	beta	399
3460	Sample_786	beta	344
3461	Sample_787	beta	437
3462	Sample_788	beta	369
3463	Sample_789	beta	368
3475	Sample_801	beta	462
3476	Sample_802	beta	285
3489	Sample_815	beta	492
3493	Sample_819	beta	314
3494	Sample_821	beta	343
3503	Sample_831	beta	387
3510	Sample_839	beta	371
3512	Sample_841	beta	292
3513	Sample_842	beta	398
3515	Sample_844	beta	580
3517	Sample_846	beta	427
3518	Sample_847	beta	641
3519	Sample_848	beta	303
3523	Sample_852	beta	469
3533	Sample_862	beta	440
3534	Sample_863	beta	307
3536	Sample_866	beta	372
3537	Sample_868	beta	422
3543	Sample_874	beta	475
3546	Sample_877	beta	419
3555	Sample_888	beta	401
3556	Sample_889	beta	313
3557	Sample_890	beta	398
3559	Sample_892	beta	309
3560	Sample_893	beta	358
3570	Sample_905	beta	389
3571	Sample_906	beta	338
3575	Sample_910	beta	479
3580	Sample_915	beta	295
3581	Sample_916	beta	432
3585	Sample_920	beta	320
3586	Sample_921	beta	386
3594	Sample_929	beta	424
3595	Sample_930	beta	420
3608	Sample_943	beta	343
3611	Sample_946	beta	452
3612	Sample_947	beta	481
3618	Sample_953	alpha	498
3629	Sample_964	alpha	491
3630	Sample_965	alpha	544
3635	Sample_970	alpha	388
3641	Sample_977	alpha	477
3642	Sample_978	alpha	422
3644	Sample_980	alpha	453
3648	Sample_984	alpha	486
3651	Sample_988	alpha	386
3658	Sample_995	alpha	468
3660	Sample_997	alpha	571
3668	Sample_1006	alpha	429
3669	Sample_1007	alpha	404
3671	Sample_1009	alpha	376
3675	Sample_1013	alpha	360
3681	Sample_1020	alpha	290
3693	Sample_1034	alpha	326
3695	Sample_1036	alpha	350
3697	Sample_1038	alpha	331
3698	Sample_1039	alpha	366
3700	Sample_1041	alpha	380
3702	Sample_1043	alpha	426
3703	Sample_1044	alpha	448
3704	Sample_1046	alpha	229
3708	Sample_1051	alpha	379
3714	Sample_1057	alpha	569
3718	Sample_1061	alpha	517
3727	Sample_1070	alpha	266
3728	Sample_1071	alpha	406
3734	Sample_1077	alpha	372
3741	Sample_1084	alpha	433
3744	Sample_1090	alpha	323
3745	Sample_1091	alpha	347
3757	Sample_1103	alpha	345
3758	Sample_1104	alpha	378
3768	Sample_1114	alpha	270
3778	Sample_1126	alpha	493
3783	Sample_1131	alpha	328
3785	Sample_1135	alpha	373
3790	Sample_1142	alpha	358
3799	Sample_1151	alpha	283
3801	Sample_1153	alpha	337
3802	Sample_1154	alpha	307
3809	Sample_1161	alpha	522
3815	Sample_1167	alpha	334
3817	Sample_1169	alpha	471
3819	Sample_1171	alpha	335
3825	Sample_1177	alpha	332
3830	Sample_1182	alpha	274
3845	Sample_1199	alpha	571
3848	Sample_1202	alpha	477
3853	Sample_1207	alpha	634
3862	Sample_1216	alpha	411
3866	Sample_1220	alpha	469
3867	Sample_1221	alpha	531
3875	Sample_1229	alpha	204
3882	Sample_1236	alpha	320
3887	Sample_1241	alpha	409
3895	Sample_1249	alpha	366
3896	Sample_1250	alpha	471
3907	Sample_1262	alpha	473
3911	Sample_1266	alpha	322
3916	Sample_1272	alpha	468
3920	Sample_1276	alpha	421
3921	Sample_1277	alpha	490
3925	Sample_1281	alpha	473
3936	Sample_1292	alpha	462
3949	Sample_1307	alpha	327
3962	Sample_1321	alpha	546
3968	Sample_1327	alpha	365
3970	Sample_1329	alpha	405
3974	Sample_1333	alpha	449
3983	Sample_1344	alpha	357
3985	Sample_1347	alpha	373
3987	Sample_1349	alpha	460
3989	Sample_1351	alpha	476
3990	Sample_1352	alpha	517
3994	Sample_1356	alpha	412
4002	Sample_1364	alpha	398
4003	Sample_1365	alpha	586
4008	Sample_1370	alpha	402
4010	Sample_1372	alpha	467
4016	Sample_1379	alpha	480
4020	Sample_1383	alpha	373
4022	Sample_1385	alpha	424
4026	Sample_1389	alpha	431
4030	Sample_1393	alpha	354
4031	Sample_1394	alpha	481
4032	Sample_1395	alpha	559
4034	Sample_1397	alpha	465
4044	Sample_1408	alpha	398
4049	Sample_1414	alpha	415
4054	Sample_1420	alpha	445
4055	Sample_1421	alpha	397
4057	Sample_1423	alpha	327
4060	Sample_1426	alpha	377
4063	Sample_1429	alpha	353
4072	Sample_1438	alpha	390
4076	Sample_1442	alpha	420
4079	Sample_1445	alpha	338
4081	Sample_1447	alpha	376
4082	Sample_1448	alpha	216
4083	Sample_1449	alpha	275
4084	Sample_1450	alpha	233
4087	Sample_1453	alpha	398
4094	Sample_1461	alpha	393
4097	Sample_1464	alpha	469
4098	Sample_1465	alpha	351
4101	Sample_1468	alpha	345
4102	Sample_1469	alpha	294
4116	Sample_1484	alpha	400
4119	Sample_1488	alpha	413
4120	Sample_1489	alpha	266
4122	Sample_1491	alpha	310
4130	Sample_1500	alpha	291
4134	Sample_1506	alpha	425
4142	Sample_1514	alpha	418
4143	Sample_1515	alpha	438
4150	Sample_1525	delta	449
4153	Sample_1528	delta	369
4162	Sample_1540	delta	250
4166	Sample_1547	PP	502
4168	Sample_1549	PP	509
4176	Sample_1557	PP	408
4181	Sample_1562	PP	400
4192	Sample_1573	PP	393
4197	Sample_1578	PP	370
4208	Sample_1589	PP	602
```
<!-- END_RESTORE_FILE -->


</details>

## file-9d6432fdf2a8

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/input_length_report.json` · SHA256: `6f8cb6e8bdd2bf27fc2f3da92fdd18eae5636be08ad7f1bf18e756a953f9e5ae`

<details>
<summary>input_length_report.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/input_length_report.json", "sha256": "6f8cb6e8bdd2bf27fc2f3da92fdd18eae5636be08ad7f1bf18e756a953f9e5ae", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "length_caps_including_cls": [
    256,
    512,
    1024,
    1200
  ],
  "pad_token_id": 60694,
  "checkpoint_sha256": "6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922",
  "input_files": {
    "test": {
      "path": "/workspace/scgpt/data/pancreas_demo/demo_test.h5ad",
      "sha256": "6fd33b1d8e5a9b3eb515ce3252869e32db7e5088edf8b9302b0ab698fdab5b29"
    },
    "train": {
      "path": "/workspace/scgpt/data/pancreas_demo/demo_train.h5ad",
      "sha256": "660e120cb991d47e101ef1259f763bdd64f2de0a51f1778336cc4322458c94e2"
    }
  },
  "tasks": {
    "inference": {
      "source_split": "test",
      "source_shape": [
        4218,
        3000
      ],
      "selected_cells": 1000,
      "matched_genes": 2999,
      "cell_order_original_row_indices": [
        18,
        19,
        22,
        23,
        24,
        26,
        30,
        34,
        48,
        49,
        50,
        59,
        64,
        65,
        67,
        76,
        79,
        91,
        96,
        103,
        114,
        116,
        117,
        124,
        129,
        132,
        136,
        143,
        144,
        146,
        151,
        160,
        161,
        164,
        174,
        177,
        186,
        196,
        197,
        200,
        207,
        208,
        210,
        217,
        221,
        224,
        227,
        233,
        234,
        243,
        248,
        250,
        264,
        272,
        277,
        280,
        287,
        289,
        292,
        295,
        299,
        303,
        304,
        305,
        307,
        321,
        325,
        327,
        329,
        331,
        339,
        342,
        346,
        352,
        354,
        359,
        365,
        370,
        372,
        377,
        378,
        381,
        382,
        383,
        386,
        389,
        393,
        399,
        403,
        405,
        410,
        414,
        415,
        419,
        422,
        426,
        427,
        429,
        433,
        438,
        441,
        442,
        445,
        452,
        461,
        463,
        465,
        466,
        471,
        473,
        478,
        484,
        485,
        487,
        490,
        491,
        494,
        496,
        501,
        502,
        505,
        507,
        508,
        511,
        513,
        514,
        516,
        517,
        521,
        528,
        530,
        535,
        536,
        538,
        541,
        544,
        546,
        547,
        548,
        551,
        553,
        557,
        558,
        564,
        565,
        576,
        579,
        584,
        586,
        587,
        591,
        593,
        601,
        609,
        611,
        616,
        620,
        622,
        623,
        625,
        632,
        634,
        635,
        636,
        641,
        644,
        647,
        650,
        653,
        656,
        657,
        661,
        663,
        669,
        671,
        685,
        686,
        696,
        700,
        704,
        710,
        714,
        719,
        721,
        727,
        732,
        738,
        747,
        757,
        783,
        785,
        787,
        792,
        794,
        800,
        803,
        813,
        815,
        819,
        823,
        825,
        829,
        833,
        838,
        861,
        863,
        867,
        882,
        893,
        895,
        901,
        903,
        907,
        910,
        912,
        914,
        922,
        923,
        927,
        929,
        934,
        937,
        944,
        946,
        948,
        955,
        957,
        965,
        971,
        975,
        989,
        991,
        993,
        995,
        997,
        1000,
        1001,
        1014,
        1019,
        1020,
        1022,
        1027,
        1028,
        1032,
        1035,
        1036,
        1044,
        1050,
        1052,
        1053,
        1065,
        1067,
        1068,
        1070,
        1071,
        1076,
        1078,
        1083,
        1087,
        1100,
        1101,
        1112,
        1113,
        1117,
        1122,
        1127,
        1132,
        1134,
        1144,
        1145,
        1149,
        1156,
        1158,
        1159,
        1160,
        1161,
        1164,
        1166,
        1169,
        1172,
        1185,
        1190,
        1192,
        1199,
        1201,
        1202,
        1204,
        1209,
        1213,
        1217,
        1222,
        1234,
        1235,
        1238,
        1241,
        1247,
        1249,
        1251,
        1252,
        1260,
        1266,
        1268,
        1270,
        1279,
        1283,
        1297,
        1300,
        1305,
        1316,
        1319,
        1325,
        1327,
        1329,
        1338,
        1342,
        1348,
        1366,
        1373,
        1374,
        1383,
        1390,
        1395,
        1409,
        1413,
        1414,
        1417,
        1436,
        1440,
        1441,
        1442,
        1443,
        1454,
        1455,
        1458,
        1460,
        1461,
        1463,
        1464,
        1466,
        1467,
        1472,
        1473,
        1475,
        1483,
        1493,
        1500,
        1518,
        1519,
        1520,
        1523,
        1524,
        1527,
        1529,
        1531,
        1532,
        1536,
        1542,
        1543,
        1545,
        1554,
        1557,
        1565,
        1566,
        1567,
        1568,
        1572,
        1574,
        1577,
        1583,
        1584,
        1588,
        1594,
        1597,
        1601,
        1616,
        1620,
        1629,
        1632,
        1634,
        1653,
        1660,
        1661,
        1664,
        1670,
        1673,
        1680,
        1683,
        1685,
        1686,
        1690,
        1691,
        1693,
        1697,
        1700,
        1701,
        1702,
        1703,
        1705,
        1708,
        1710,
        1714,
        1718,
        1720,
        1725,
        1730,
        1734,
        1739,
        1740,
        1753,
        1771,
        1772,
        1776,
        1777,
        1779,
        1785,
        1789,
        1790,
        1803,
        1808,
        1814,
        1817,
        1821,
        1825,
        1829,
        1830,
        1832,
        1834,
        1835,
        1836,
        1838,
        1840,
        1841,
        1842,
        1849,
        1850,
        1854,
        1857,
        1859,
        1861,
        1862,
        1864,
        1865,
        1868,
        1874,
        1880,
        1885,
        1890,
        1892,
        1893,
        1895,
        1896,
        1901,
        1902,
        1904,
        1910,
        1912,
        1914,
        1919,
        1923,
        1926,
        1937,
        1940,
        1942,
        1943,
        1953,
        1955,
        1967,
        1968,
        1973,
        1978,
        1987,
        1988,
        1995,
        2000,
        2004,
        2020,
        2022,
        2024,
        2027,
        2029,
        2037,
        2039,
        2040,
        2043,
        2049,
        2053,
        2057,
        2061,
        2063,
        2065,
        2069,
        2071,
        2073,
        2080,
        2088,
        2095,
        2108,
        2110,
        2112,
        2113,
        2116,
        2125,
        2135,
        2141,
        2148,
        2149,
        2162,
        2163,
        2164,
        2167,
        2169,
        2176,
        2177,
        2181,
        2186,
        2192,
        2193,
        2197,
        2200,
        2205,
        2209,
        2210,
        2211,
        2214,
        2216,
        2218,
        2221,
        2226,
        2230,
        2231,
        2232,
        2233,
        2238,
        2239,
        2243,
        2244,
        2245,
        2246,
        2247,
        2249,
        2259,
        2260,
        2262,
        2263,
        2284,
        2291,
        2294,
        2298,
        2310,
        2311,
        2313,
        2318,
        2319,
        2321,
        2330,
        2342,
        2346,
        2349,
        2350,
        2353,
        2355,
        2361,
        2364,
        2368,
        2369,
        2373,
        2375,
        2377,
        2389,
        2394,
        2396,
        2397,
        2402,
        2415,
        2420,
        2422,
        2425,
        2428,
        2433,
        2434,
        2436,
        2441,
        2442,
        2446,
        2450,
        2452,
        2453,
        2458,
        2460,
        2461,
        2462,
        2464,
        2467,
        2471,
        2475,
        2476,
        2484,
        2485,
        2492,
        2513,
        2515,
        2517,
        2520,
        2525,
        2527,
        2529,
        2534,
        2537,
        2539,
        2542,
        2543,
        2548,
        2550,
        2559,
        2563,
        2564,
        2570,
        2581,
        2588,
        2592,
        2594,
        2597,
        2599,
        2600,
        2601,
        2609,
        2615,
        2616,
        2617,
        2618,
        2623,
        2624,
        2627,
        2630,
        2633,
        2638,
        2642,
        2647,
        2648,
        2650,
        2654,
        2661,
        2663,
        2664,
        2665,
        2668,
        2669,
        2670,
        2673,
        2678,
        2685,
        2687,
        2689,
        2694,
        2695,
        2697,
        2698,
        2707,
        2718,
        2719,
        2722,
        2723,
        2728,
        2736,
        2742,
        2748,
        2749,
        2753,
        2754,
        2756,
        2759,
        2760,
        2763,
        2766,
        2768,
        2769,
        2778,
        2782,
        2785,
        2786,
        2789,
        2793,
        2795,
        2814,
        2820,
        2823,
        2825,
        2826,
        2833,
        2836,
        2845,
        2851,
        2854,
        2858,
        2860,
        2865,
        2870,
        2872,
        2873,
        2878,
        2882,
        2888,
        2889,
        2898,
        2902,
        2904,
        2908,
        2916,
        2917,
        2921,
        2926,
        2931,
        2947,
        2948,
        2949,
        2955,
        2968,
        2969,
        2974,
        2975,
        2986,
        2993,
        2996,
        3003,
        3004,
        3014,
        3015,
        3019,
        3024,
        3026,
        3047,
        3049,
        3052,
        3057,
        3058,
        3059,
        3060,
        3068,
        3082,
        3085,
        3091,
        3093,
        3099,
        3102,
        3103,
        3107,
        3108,
        3114,
        3123,
        3126,
        3127,
        3129,
        3137,
        3143,
        3144,
        3149,
        3150,
        3151,
        3156,
        3158,
        3160,
        3167,
        3168,
        3175,
        3180,
        3184,
        3186,
        3188,
        3189,
        3192,
        3200,
        3212,
        3215,
        3217,
        3219,
        3221,
        3226,
        3228,
        3234,
        3242,
        3244,
        3262,
        3266,
        3275,
        3277,
        3278,
        3282,
        3286,
        3287,
        3292,
        3298,
        3301,
        3304,
        3308,
        3311,
        3313,
        3320,
        3330,
        3334,
        3337,
        3338,
        3344,
        3347,
        3348,
        3363,
        3365,
        3372,
        3375,
        3376,
        3379,
        3384,
        3388,
        3390,
        3391,
        3392,
        3394,
        3400,
        3404,
        3412,
        3413,
        3423,
        3424,
        3429,
        3439,
        3440,
        3442,
        3443,
        3447,
        3451,
        3454,
        3457,
        3460,
        3461,
        3462,
        3463,
        3475,
        3476,
        3489,
        3493,
        3494,
        3503,
        3510,
        3512,
        3513,
        3515,
        3517,
        3518,
        3519,
        3523,
        3533,
        3534,
        3536,
        3537,
        3543,
        3546,
        3555,
        3556,
        3557,
        3559,
        3560,
        3570,
        3571,
        3575,
        3580,
        3581,
        3585,
        3586,
        3594,
        3595,
        3608,
        3611,
        3612,
        3618,
        3629,
        3630,
        3635,
        3641,
        3642,
        3644,
        3648,
        3651,
        3658,
        3660,
        3668,
        3669,
        3671,
        3675,
        3681,
        3693,
        3695,
        3697,
        3698,
        3700,
        3702,
        3703,
        3704,
        3708,
        3714,
        3718,
        3727,
        3728,
        3734,
        3741,
        3744,
        3745,
        3757,
        3758,
        3768,
        3778,
        3783,
        3785,
        3790,
        3799,
        3801,
        3802,
        3809,
        3815,
        3817,
        3819,
        3825,
        3830,
        3845,
        3848,
        3853,
        3862,
        3866,
        3867,
        3875,
        3882,
        3887,
        3895,
        3896,
        3907,
        3911,
        3916,
        3920,
        3921,
        3925,
        3936,
        3949,
        3962,
        3968,
        3970,
        3974,
        3983,
        3985,
        3987,
        3989,
        3990,
        3994,
        4002,
        4003,
        4008,
        4010,
        4016,
        4020,
        4022,
        4026,
        4030,
        4031,
        4032,
        4034,
        4044,
        4049,
        4054,
        4055,
        4057,
        4060,
        4063,
        4072,
        4076,
        4079,
        4081,
        4082,
        4083,
        4084,
        4087,
        4094,
        4097,
        4098,
        4101,
        4102,
        4116,
        4119,
        4120,
        4122,
        4130,
        4134,
        4142,
        4143,
        4150,
        4153,
        4162,
        4166,
        4168,
        4176,
        4181,
        4192,
        4197,
        4208
      ],
      "tokens_before_sampling_including_cls": {
        "min": 185,
        "mean": 616.229,
        "median": 558.0,
        "p95": 1066.1,
        "max": 1302
      },
      "caps": {
        "256": {
          "actual_nonpadding_tokens": {
            "min": 185,
            "mean": 255.667,
            "median": 256.0,
            "p95": 256.0,
            "max": 256
          },
          "batch_padded_tokens": {
            "min": 256,
            "mean": 256.0,
            "median": 256.0,
            "p95": 256.0,
            "max": 256
          },
          "cells_reaching_cap": 989,
          "cells_shortened": 989,
          "batches_reaching_cap": 125
        },
        "512": {
          "actual_nonpadding_tokens": {
            "min": 185,
            "mean": 459.925,
            "median": 512.0,
            "p95": 512.0,
            "max": 512
          },
          "batch_padded_tokens": {
            "min": 412,
            "mean": 504.192,
            "median": 512.0,
            "p95": 512.0,
            "max": 512
          },
          "cells_reaching_cap": 555,
          "cells_shortened": 554,
          "batches_reaching_cap": 103
        },
        "1024": {
          "actual_nonpadding_tokens": {
            "min": 185,
            "mean": 610.885,
            "median": 558.0,
            "p95": 1024.0,
            "max": 1024
          },
          "batch_padded_tokens": {
            "min": 412,
            "mean": 796.016,
            "median": 897.0,
            "p95": 1024.0,
            "max": 1024
          },
          "cells_reaching_cap": 85,
          "cells_shortened": 83,
          "batches_reaching_cap": 50
        },
        "1200": {
          "actual_nonpadding_tokens": {
            "min": 185,
            "mean": 616.004,
            "median": 558.0,
            "p95": 1066.1,
            "max": 1200
          },
          "batch_padded_tokens": {
            "min": 412,
            "mean": 825.56,
            "median": 897.0,
            "p95": 1167.0,
            "max": 1200
          },
          "cells_reaching_cap": 4,
          "cells_shortened": 4,
          "batches_reaching_cap": 4
        }
      }
    },
    "finetune": {
      "source_split": "train",
      "source_shape": [
        10600,
        3000
      ],
      "selected_cells": 1000,
      "matched_genes": 2999,
      "cell_order_original_row_indices": [
        10264,
        7463,
        2620,
        2530,
        8758,
        7420,
        4315,
        2328,
        4264,
        3622,
        2789,
        5273,
        2374,
        4199,
        8059,
        7668,
        3050,
        1665,
        3276,
        2396,
        7324,
        34,
        5792,
        8715,
        5926,
        3418,
        3555,
        3487,
        8067,
        8467,
        9936,
        5306,
        2497,
        9741,
        9828,
        5859,
        880,
        3999,
        9584,
        3387,
        5265,
        10565,
        9312,
        4498,
        10491,
        5359,
        9927,
        3161,
        8739,
        5343,
        6852,
        10297,
        4464,
        560,
        3431,
        5771,
        8083,
        1341,
        7359,
        2019,
        2851,
        10421,
        6539,
        414,
        7019,
        4836,
        4276,
        6905,
        5248,
        3693,
        2898,
        4853,
        5349,
        1879,
        2087,
        2529,
        4747,
        5016,
        1751,
        7117,
        10529,
        9206,
        9587,
        1333,
        2420,
        6076,
        5275,
        1772,
        10170,
        3786,
        845,
        6718,
        742,
        9124,
        3782,
        7258,
        1529,
        7983,
        7111,
        4465,
        4850,
        4905,
        9907,
        2385,
        6332,
        7454,
        1171,
        10424,
        503,
        1176,
        1362,
        8419,
        4516,
        7517,
        7060,
        4327,
        1632,
        1289,
        255,
        6505,
        942,
        3235,
        3775,
        2912,
        9302,
        3769,
        4084,
        8600,
        5085,
        8842,
        1149,
        1593,
        1615,
        2023,
        5597,
        315,
        10072,
        3590,
        8370,
        6459,
        2002,
        7972,
        4964,
        9973,
        2127,
        1721,
        3312,
        6305,
        5337,
        4060,
        7354,
        5363,
        6764,
        8374,
        8798,
        5330,
        2501,
        5867,
        1420,
        6422,
        7882,
        6091,
        3842,
        10364,
        8475,
        10543,
        724,
        7598,
        10047,
        5297,
        9854,
        5576,
        3720,
        7523,
        10518,
        8557,
        7613,
        8152,
        9845,
        8305,
        5407,
        7659,
        7278,
        6532,
        5405,
        3690,
        4104,
        6771,
        7848,
        8603,
        5244,
        9062,
        18,
        480,
        438,
        5936,
        2904,
        648,
        6362,
        4890,
        5546,
        9148,
        5319,
        1501,
        2683,
        4288,
        8723,
        8108,
        7963,
        725,
        2949,
        6940,
        10000,
        6853,
        40,
        9751,
        6277,
        9273,
        1210,
        2051,
        2749,
        5370,
        364,
        1844,
        9755,
        10269,
        7518,
        458,
        5328,
        1372,
        402,
        1009,
        4763,
        2710,
        5209,
        7053,
        8302,
        4936,
        9829,
        9140,
        7328,
        7700,
        331,
        1380,
        3297,
        7201,
        6906,
        581,
        1276,
        6405,
        694,
        2663,
        894,
        8091,
        8731,
        10148,
        9172,
        4912,
        10391,
        4245,
        633,
        6115,
        2326,
        6602,
        728,
        3994,
        5779,
        4616,
        5993,
        9147,
        8384,
        1402,
        10092,
        6084,
        6549,
        7872,
        7419,
        8730,
        3647,
        2849,
        689,
        3997,
        3638,
        9890,
        7938,
        9418,
        9055,
        4520,
        8403,
        1114,
        6470,
        7035,
        4864,
        9364,
        2577,
        6072,
        5719,
        9109,
        4668,
        61,
        4706,
        5409,
        4600,
        9165,
        229,
        2504,
        1239,
        5261,
        10445,
        7749,
        6268,
        6358,
        2926,
        4988,
        7159,
        6784,
        9385,
        7330,
        1596,
        5267,
        1503,
        2692,
        7195,
        8360,
        9582,
        6727,
        3686,
        6479,
        6935,
        4005,
        3619,
        6133,
        9964,
        266,
        1666,
        9744,
        10243,
        8373,
        9851,
        5728,
        3325,
        9743,
        3211,
        1924,
        1044,
        9753,
        2011,
        3815,
        10566,
        4399,
        5620,
        2918,
        4260,
        2339,
        5937,
        7708,
        2251,
        6559,
        970,
        2150,
        5441,
        10374,
        4915,
        9197,
        7690,
        8550,
        3368,
        10580,
        5797,
        3030,
        1153,
        3948,
        5967,
        6015,
        9822,
        10318,
        9015,
        1604,
        752,
        6028,
        174,
        2936,
        3770,
        9024,
        1491,
        10284,
        5798,
        5377,
        4129,
        6356,
        4589,
        5430,
        7573,
        3377,
        2352,
        5997,
        7982,
        6497,
        2187,
        9560,
        10220,
        7028,
        593,
        8814,
        4340,
        7658,
        9283,
        2422,
        2612,
        8502,
        8489,
        9799,
        8913,
        2086,
        3845,
        10417,
        2942,
        3071,
        2358,
        10200,
        1589,
        7549,
        6443,
        7240,
        1108,
        9848,
        1552,
        635,
        5791,
        5538,
        1450,
        7076,
        7893,
        1222,
        1517,
        6101,
        1466,
        10108,
        7460,
        228,
        4877,
        7193,
        3569,
        4238,
        9307,
        6216,
        9404,
        2516,
        2865,
        10420,
        10001,
        3132,
        4153,
        1374,
        1356,
        4495,
        6528,
        4686,
        7261,
        6436,
        1738,
        2517,
        5679,
        2278,
        797,
        5236,
        932,
        9568,
        6974,
        7827,
        5563,
        4943,
        4594,
        6933,
        2340,
        793,
        116,
        79,
        1095,
        7590,
        8718,
        3163,
        270,
        10449,
        10096,
        8621,
        7095,
        8310,
        337,
        82,
        2190,
        5558,
        5710,
        8979,
        9199,
        4602,
        8948,
        253,
        2956,
        5803,
        6141,
        224,
        7563,
        9937,
        5284,
        7384,
        1115,
        7576,
        5336,
        3888,
        9608,
        8929,
        940,
        1661,
        2527,
        10539,
        1232,
        6911,
        1590,
        5140,
        4986,
        7464,
        1026,
        6383,
        2495,
        10053,
        89,
        1785,
        736,
        8970,
        9529,
        6281,
        7702,
        7398,
        3432,
        978,
        4951,
        3017,
        1762,
        8517,
        9819,
        3181,
        10494,
        9656,
        4330,
        7741,
        8301,
        3450,
        5455,
        6713,
        6012,
        6592,
        6502,
        3882,
        7176,
        5898,
        3737,
        3375,
        110,
        3741,
        2614,
        1594,
        7392,
        6560,
        4644,
        5640,
        9814,
        8581,
        8259,
        7435,
        8251,
        5702,
        7959,
        5958,
        470,
        6162,
        3121,
        4316,
        630,
        7960,
        3055,
        6621,
        218,
        7309,
        8194,
        5861,
        6903,
        9112,
        1020,
        10182,
        2662,
        10427,
        4710,
        4233,
        10123,
        8472,
        4460,
        6789,
        7794,
        1361,
        6071,
        825,
        2561,
        1734,
        9598,
        4292,
        1541,
        2133,
        3730,
        1640,
        10271,
        5218,
        206,
        1780,
        7492,
        4844,
        3099,
        8911,
        7551,
        10464,
        1396,
        3044,
        6253,
        7682,
        6346,
        8838,
        1363,
        4578,
        8835,
        3059,
        3538,
        1446,
        4574,
        8239,
        7575,
        9028,
        5864,
        4628,
        7688,
        9777,
        2651,
        8905,
        6806,
        3725,
        9454,
        6962,
        2508,
        4052,
        2080,
        1606,
        6732,
        4554,
        6345,
        1479,
        2854,
        9940,
        10354,
        887,
        7651,
        7861,
        4824,
        7847,
        8857,
        3434,
        360,
        6872,
        5615,
        443,
        1349,
        1988,
        8755,
        9949,
        75,
        8013,
        906,
        6282,
        8866,
        5063,
        9688,
        9001,
        10254,
        1069,
        3979,
        211,
        7596,
        4147,
        841,
        1265,
        10337,
        6677,
        2390,
        2546,
        1481,
        4241,
        83,
        9566,
        1856,
        4058,
        2500,
        7357,
        6730,
        4206,
        1828,
        4963,
        2320,
        2089,
        1234,
        9222,
        7990,
        856,
        6176,
        4113,
        1245,
        8951,
        1808,
        4452,
        7058,
        7602,
        6683,
        8633,
        848,
        6271,
        7319,
        6681,
        10332,
        3881,
        5755,
        4013,
        5557,
        2709,
        8578,
        8330,
        3809,
        2563,
        448,
        6793,
        7347,
        8858,
        2951,
        2140,
        1391,
        8248,
        6896,
        4533,
        4565,
        2803,
        3898,
        4064,
        351,
        7027,
        3183,
        10460,
        4457,
        7820,
        2265,
        10545,
        9114,
        4424,
        4728,
        578,
        7360,
        5176,
        7129,
        9004,
        5131,
        5872,
        8232,
        2094,
        4072,
        7049,
        8810,
        966,
        10480,
        8087,
        3479,
        8174,
        7973,
        7707,
        6715,
        4442,
        5220,
        6863,
        10388,
        9546,
        9371,
        4165,
        4849,
        5485,
        2771,
        4086,
        4612,
        5446,
        4960,
        4711,
        3112,
        4647,
        4271,
        6844,
        5257,
        8536,
        9803,
        5467,
        568,
        3516,
        6918,
        4111,
        8000,
        6904,
        9352,
        7971,
        8256,
        7746,
        1301,
        8670,
        2841,
        7986,
        6653,
        6657,
        10294,
        7115,
        6827,
        9116,
        4198,
        6255,
        8790,
        3671,
        5415,
        3144,
        9796,
        9733,
        520,
        7022,
        5355,
        8159,
        1223,
        5457,
        3080,
        7124,
        10074,
        7851,
        9193,
        775,
        10126,
        4566,
        10552,
        656,
        4699,
        951,
        2698,
        1231,
        2866,
        5639,
        3599,
        4354,
        6558,
        7939,
        2678,
        84,
        7074,
        7937,
        9840,
        10063,
        2693,
        579,
        6415,
        7155,
        7902,
        9025,
        2766,
        584,
        9739,
        2003,
        2164,
        5047,
        9031,
        10397,
        9274,
        7574,
        1523,
        4414,
        334,
        1767,
        157,
        6210,
        7355,
        652,
        7446,
        9525,
        2984,
        8681,
        5855,
        5599,
        6787,
        1154,
        194,
        5782,
        1660,
        9838,
        5846,
        1740,
        6287,
        6938,
        7125,
        2969,
        7296,
        7021,
        8620,
        904,
        8533,
        3593,
        2066,
        8004,
        5788,
        2050,
        152,
        10078,
        2143,
        8626,
        4275,
        10281,
        6464,
        6884,
        8812,
        9524,
        8795,
        6825,
        955,
        10508,
        303,
        4758,
        5857,
        10060,
        10573,
        9897,
        9921,
        5460,
        9458,
        7987,
        7344,
        6908,
        6132,
        1185,
        6740,
        9289,
        9783,
        585,
        3812,
        5884,
        2330,
        600,
        2512,
        571,
        4473,
        7406,
        10381,
        9490,
        6832,
        3909,
        6145,
        1622,
        6566,
        3853,
        7133,
        2885,
        7380,
        10500,
        2134,
        3953,
        2438,
        5929,
        5682,
        5157,
        5837,
        9643,
        6831,
        5345,
        7969,
        8565,
        8203,
        6834,
        8046,
        7452,
        301,
        5231,
        2657,
        1811,
        3834,
        789,
        552,
        858,
        4935,
        5515,
        7468,
        9111,
        1911,
        5617,
        192,
        1944,
        4614,
        727
      ],
      "tokens_before_sampling_including_cls": {
        "min": 72,
        "mean": 310.597,
        "median": 244.0,
        "p95": 710.6499999999994,
        "max": 1244
      },
      "caps": {
        "256": {
          "actual_nonpadding_tokens": {
            "min": 72,
            "mean": 220.004,
            "median": 244.0,
            "p95": 256.0,
            "max": 256
          },
          "batch_padded_tokens": {
            "min": 256,
            "mean": 256.0,
            "median": 256.0,
            "p95": 256.0,
            "max": 256
          },
          "cells_reaching_cap": 468,
          "cells_shortened": 466,
          "batches_reaching_cap": 125
        },
        "512": {
          "actual_nonpadding_tokens": {
            "min": 72,
            "mean": 286.681,
            "median": 244.0,
            "p95": 512.0,
            "max": 512
          },
          "batch_padded_tokens": {
            "min": 268,
            "mean": 474.232,
            "median": 512.0,
            "p95": 512.0,
            "max": 512
          },
          "cells_reaching_cap": 123,
          "cells_shortened": 123,
          "batches_reaching_cap": 77
        },
        "1024": {
          "actual_nonpadding_tokens": {
            "min": 72,
            "mean": 309.995,
            "median": 244.0,
            "p95": 710.6499999999994,
            "max": 1024
          },
          "batch_padded_tokens": {
            "min": 268,
            "mean": 612.536,
            "median": 558.0,
            "p95": 991.6,
            "max": 1024
          },
          "cells_reaching_cap": 6,
          "cells_shortened": 6,
          "batches_reaching_cap": 5
        },
        "1200": {
          "actual_nonpadding_tokens": {
            "min": 72,
            "mean": 310.553,
            "median": 244.0,
            "p95": 710.6499999999994,
            "max": 1200
          },
          "batch_padded_tokens": {
            "min": 268,
            "mean": 615.896,
            "median": 558.0,
            "p95": 991.6,
            "max": 1200
          },
          "cells_reaching_cap": 1,
          "cells_shortened": 1,
          "batches_reaching_cap": 1
        }
      }
    }
  }
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-86a1e5857cc4

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/process_records.json` · SHA256: `761f016f891438558556a98045afac892a94f16a111767a11784354048b3443f`

<details>
<summary>process_records.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/process_records.json", "sha256": "761f016f891438558556a98045afac892a94f16a111767a11784354048b3443f", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "inference_L0256": {
    "returncode": 0,
    "whole_process_seconds": 101.44052228494547
  },
  "inference_L0512": {
    "returncode": 0,
    "whole_process_seconds": 105.35337310284376
  },
  "inference_L1024": {
    "returncode": 0,
    "whole_process_seconds": 98.20881273387931
  },
  "inference_L1200": {
    "returncode": 0,
    "whole_process_seconds": 107.93387867789716
  },
  "finetune_L0256": {
    "returncode": 0,
    "whole_process_seconds": 115.47962304390967
  },
  "finetune_L0512": {
    "returncode": 0,
    "whole_process_seconds": 121.79608216392808
  },
  "finetune_L1024": {
    "returncode": 0,
    "whole_process_seconds": 123.01876717992127
  },
  "finetune_L1200": {
    "returncode": 0,
    "whole_process_seconds": 122.77139263506979
  }
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-99e214e5404a

원래 경로: `scgpt/results/length_compare_20261023_040313_501423/summary.json` · SHA256: `20ab82fcc7ea4e54e4a0595abb46788daaa4e4712ef3838b281626cf93d8059a`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/summary.json", "sha256": "20ab82fcc7ea4e54e4a0595abb46788daaa4e4712ef3838b281626cf93d8059a", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "status": "complete",
  "completed_cases": 8,
  "expected_cases": 8,
  "measurements": [
    {
      "task": "inference",
      "source_split": "test",
      "cells": 1000,
      "max_length_including_cls": 256,
      "actual_tokens_min": 185,
      "actual_tokens_mean": 255.667,
      "actual_tokens_max": 256,
      "batch_padded_tokens_mean": 256.0,
      "batch_padded_tokens_max": 256,
      "time_seconds": 1.2901826270390302,
      "cells_per_second": 775.0840687531195,
      "gpu_peak_allocated_gib": 0.25962162017822266,
      "gpu_peak_reserved_gib": 0.28125,
      "process_rss_peak_observed_gib": 2.0585098266601562,
      "gpu_allocation_pct_of_system_ram": 0.2133510735477616,
      "system_total_memory_gib": 121.68751525878906,
      "whole_process_seconds": 101.44052228494547,
      "same_cells_and_order_verified": true,
      "checkpoint_and_input_hashes_verified": true,
      "saved_model_reload_verified": null,
      "model_bundle": null,
      "timing_scope": "gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording",
      "results_directory": "/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L0256"
    },
    {
      "task": "inference",
      "source_split": "test",
      "cells": 1000,
      "max_length_including_cls": 512,
      "actual_tokens_min": 185,
      "actual_tokens_mean": 459.925,
      "actual_tokens_max": 512,
      "batch_padded_tokens_mean": 504.192,
      "batch_padded_tokens_max": 512,
      "time_seconds": 2.4339966219849885,
      "cells_per_second": 410.8469136594255,
      "gpu_peak_allocated_gib": 0.29481029510498047,
      "gpu_peak_reserved_gib": 0.3125,
      "process_rss_peak_observed_gib": 1.7441253662109375,
      "gpu_allocation_pct_of_system_ram": 0.24226831690828476,
      "system_total_memory_gib": 121.68751525878906,
      "whole_process_seconds": 105.35337310284376,
      "same_cells_and_order_verified": true,
      "checkpoint_and_input_hashes_verified": true,
      "saved_model_reload_verified": null,
      "model_bundle": null,
      "timing_scope": "gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording",
      "results_directory": "/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L0512"
    },
    {
      "task": "inference",
      "source_split": "test",
      "cells": 1000,
      "max_length_including_cls": 1024,
      "actual_tokens_min": 185,
      "actual_tokens_mean": 610.885,
      "actual_tokens_max": 1024,
      "batch_padded_tokens_mean": 796.016,
      "batch_padded_tokens_max": 1024,
      "time_seconds": 3.531116613186896,
      "cells_per_second": 283.1965379635203,
      "gpu_peak_allocated_gib": 0.3615903854370117,
      "gpu_peak_reserved_gib": 0.41796875,
      "process_rss_peak_observed_gib": 2.047393798828125,
      "gpu_allocation_pct_of_system_ram": 0.29714665852781086,
      "system_total_memory_gib": 121.68751525878906,
      "whole_process_seconds": 98.20881273387931,
      "same_cells_and_order_verified": true,
      "checkpoint_and_input_hashes_verified": true,
      "saved_model_reload_verified": null,
      "model_bundle": null,
      "timing_scope": "gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording",
      "results_directory": "/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L1024"
    },
    {
      "task": "inference",
      "source_split": "test",
      "cells": 1000,
      "max_length_including_cls": 1200,
      "actual_tokens_min": 185,
      "actual_tokens_mean": 616.004,
      "actual_tokens_max": 1200,
      "batch_padded_tokens_mean": 825.56,
      "batch_padded_tokens_max": 1200,
      "time_seconds": 3.588488161796704,
      "cells_per_second": 278.6688864257851,
      "gpu_peak_allocated_gib": 0.36938047409057617,
      "gpu_peak_reserved_gib": 0.51171875,
      "process_rss_peak_observed_gib": 1.7204704284667969,
      "gpu_allocation_pct_of_system_ram": 0.3035483741327335,
      "system_total_memory_gib": 121.68751525878906,
      "whole_process_seconds": 107.93387867789716,
      "same_cells_and_order_verified": true,
      "checkpoint_and_input_hashes_verified": true,
      "saved_model_reload_verified": null,
      "model_bundle": null,
      "timing_scope": "gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording",
      "results_directory": "/workspace/scgpt/results/length_compare_20261023_040313_501423/inference_L1200"
    },
    {
      "task": "finetune",
      "source_split": "train",
      "cells": 1000,
      "max_length_including_cls": 256,
      "actual_tokens_min": 72,
      "actual_tokens_mean": 220.004,
      "actual_tokens_max": 256,
      "batch_padded_tokens_mean": 256.0,
      "batch_padded_tokens_max": 256,
      "time_seconds": 9.221140660112724,
      "cells_per_second": 108.44645330329182,
      "gpu_peak_allocated_gib": 1.2221388816833496,
      "gpu_peak_reserved_gib": 1.45703125,
      "process_rss_peak_observed_gib": 2.6563873291015625,
      "gpu_allocation_pct_of_system_ram": 1.0043256114518113,
      "system_total_memory_gib": 121.68751525878906,
      "whole_process_seconds": 115.47962304390967,
      "same_cells_and_order_verified": true,
      "checkpoint_and_input_hashes_verified": true,
      "saved_model_reload_verified": true,
      "model_bundle": "/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001",
      "timing_scope": "tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization",
      "results_directory": "/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L0256"
    },
    {
      "task": "finetune",
      "source_split": "train",
      "cells": 1000,
      "max_length_including_cls": 512,
      "actual_tokens_min": 72,
      "actual_tokens_mean": 286.681,
      "actual_tokens_max": 512,
      "batch_padded_tokens_mean": 474.232,
      "batch_padded_tokens_max": 512,
      "time_seconds": 15.796376958023757,
      "cells_per_second": 63.30565563593054,
      "gpu_peak_allocated_gib": 1.8074431419372559,
      "gpu_peak_reserved_gib": 2.146484375,
      "process_rss_peak_observed_gib": 2.658599853515625,
      "gpu_allocation_pct_of_system_ram": 1.4853151846296004,
      "system_total_memory_gib": 121.68751525878906,
      "whole_process_seconds": 121.79608216392808,
      "same_cells_and_order_verified": true,
      "checkpoint_and_input_hashes_verified": true,
      "saved_model_reload_verified": true,
      "model_bundle": "/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001",
      "timing_scope": "tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization",
      "results_directory": "/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L0512"
    },
    {
      "task": "finetune",
      "source_split": "train",
      "cells": 1000,
      "max_length_including_cls": 1024,
      "actual_tokens_min": 72,
      "actual_tokens_mean": 309.995,
      "actual_tokens_max": 1024,
      "batch_padded_tokens_mean": 612.536,
      "batch_padded_tokens_max": 1024,
      "time_seconds": 21.328996536089107,
      "cells_per_second": 46.88453103304598,
      "gpu_peak_allocated_gib": 2.9719724655151367,
      "gpu_peak_reserved_gib": 3.630859375,
      "process_rss_peak_observed_gib": 2.7067222595214844,
      "gpu_allocation_pct_of_system_ram": 2.4422985868309786,
      "system_total_memory_gib": 121.68751525878906,
      "whole_process_seconds": 123.01876717992127,
      "same_cells_and_order_verified": true,
      "checkpoint_and_input_hashes_verified": true,
      "saved_model_reload_verified": true,
      "model_bundle": "/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001",
      "timing_scope": "tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization",
      "results_directory": "/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L1024"
    },
    {
      "task": "finetune",
      "source_split": "train",
      "cells": 1000,
      "max_length_including_cls": 1200,
      "actual_tokens_min": 72,
      "actual_tokens_mean": 310.553,
      "actual_tokens_max": 1200,
      "batch_padded_tokens_mean": 615.896,
      "batch_padded_tokens_max": 1200,
      "time_seconds": 21.51013845205307,
      "cells_per_second": 46.48970541166151,
      "gpu_peak_allocated_gib": 3.374143600463867,
      "gpu_peak_reserved_gib": 4.5546875,
      "process_rss_peak_observed_gib": 2.7139358520507812,
      "gpu_allocation_pct_of_system_ram": 2.7727935715411567,
      "system_total_memory_gib": 121.68751525878906,
      "whole_process_seconds": 122.77139263506979,
      "same_cells_and_order_verified": true,
      "checkpoint_and_input_hashes_verified": true,
      "saved_model_reload_verified": true,
      "model_bundle": "/workspace/scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001",
      "timing_scope": "tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization",
      "results_directory": "/workspace/scgpt/results/length_compare_20261023_040313_501423/finetune_L1200"
    }
  ],
  "design": {
    "cases": [
      {
        "name": "inference_L0256",
        "task": "inference",
        "max_length": 256
      },
      {
        "name": "inference_L0512",
        "task": "inference",
        "max_length": 512
      },
      {
        "name": "inference_L1024",
        "task": "inference",
        "max_length": 1024
      },
      {
        "name": "inference_L1200",
        "task": "inference",
        "max_length": 1200
      },
      {
        "name": "finetune_L0256",
        "task": "finetune",
        "max_length": 256
      },
      {
        "name": "finetune_L0512",
        "task": "finetune",
        "max_length": 512
      },
      {
        "name": "finetune_L1024",
        "task": "finetune",
        "max_length": 1024
      },
      {
        "name": "finetune_L1200",
        "task": "finetune",
        "max_length": 1200
      }
    ],
    "seed": 42,
    "cells_per_task": 1000,
    "batch_size": 8,
    "precision": "FP32",
    "epochs_per_finetune": 1,
    "repeats_per_condition": 1,
    "length_policy": "official random nonzero-gene sampling with dynamic batch padding; cap includes CLS",
    "inference_split": "test",
    "finetune_split": "train; stratified 20% validation held out",
    "source_sha256": {
      "01_inference.py": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c",
      "02_finetune.py": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc"
    }
  },
  "resource_notes": [
    "Maximum-length caps include CLS and are not fixed actual token lengths.",
    "Inference uses 1000 test cells; fine-tuning uses 1000 training cells. Compare length effects within each task.",
    "Cell identities, order, input hashes and initial checkpoint hashes are verified across length conditions.",
    "Official random gene sampling is unchanged; subsets are not guaranteed to be nested between caps.",
    "FP32, batch 8, one measurement per condition; no GPU utilization, temperature or power monitoring.",
    "Inference excludes one-batch warmup. Fine-tuning includes first-step initialization and excludes validation/saving.",
    "Fresh process for every case. Reserved memory may include within-case warmup caches.",
    "PyTorch CUDA allocator counters and process RSS are separate measurements on shared physical memory; do not sum them.",
    "Whole-process time includes imports, input/model load, validation, saving and reload checks where applicable."
  ]
}
```
<!-- END_RESTORE_FILE -->


</details>

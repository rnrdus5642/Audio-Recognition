**한국어 모델 전체 결과 — 2026-09-13 시작 실험**

[판단·검증 방법·오확정 결과](WAV2VEC2_KOREAN_MODEL_SELECTION_2026-09-13.md)

표의 정렬은 Common Voice 기준 소리 오류율이다. 자료 중복·출력 규칙·단어 성능을 반영한 최종 추천 순서와 다르다. 라이선스는 공개 메타데이터 또는 카드 표기이며 성능 점수가 아니다.

| 저장소 | 디코더 | 규모 | 성인 문장 오류율 ↓ | Zeroth 오류율 ↓ | 성인 단어 일치 | 악기놀이 일치 | 라이선스 표기 | 검토 사항 |
|---|---|---|---|---|---|---|---|---|
| [kresnik/wav2vec2-large-xls-r-300m-korean](https://huggingface.co/kresnik/wav2vec2-large-xls-r-300m-korean) | syllable | 316.5M | 7.4% | 29.4% | 3/7 | 3/46 | apache-2.0 | CV13 학습 이력 |
| [Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2) | compat_heard | 315.5M | 8.5% | 8.7% | 7/7 | 31/46 | apache-2.0 | — |
| [student-47/wav2vec2-large-xlrs-korean-v5](https://huggingface.co/student-47/wav2vec2-large-xlrs-korean-v5) | compat_spelling | 315.5M | 9.1% | 2.8% | 4/7 | 17/46 | apache-2.0 | — |
| [Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-semi-supervised_V2](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-semi-supervised_V2) | compat_heard | 315.5M | 9.6% | 9.9% | 7/7 | 28/46 | apache-2.0 | — |
| [thisisHJLee/wav2vec2-large-xls-r-1b-korean-sample2](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-1b-korean-sample2) | compat_spelling | 962.6M | 9.9% | 9.5% | 6/7 | 17/46 | apache-2.0 | — |
| [tlsalfm820/wav2vec2-large-xls-r-korean-all](https://huggingface.co/tlsalfm820/wav2vec2-large-xls-r-korean-all) | compat_spelling | 315.5M | 10.0% | 3.4% | 4/7 | 18/46 | apache-2.0 | — |
| [student-47/wav2vec2-large-xlrs-korean-v4](https://huggingface.co/student-47/wav2vec2-large-xlrs-korean-v4) | compat_spelling | 315.5M | 10.4% | 3.8% | 3/7 | 17/46 | apache-2.0 | — |
| [Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based) | compat_heard | 315.5M | 10.7% | 11.0% | 3/7 | 29/46 | apache-2.0 | — |
| [lsnoo/XLS-R-300m_kspon_jamo](https://huggingface.co/lsnoo/XLS-R-300m_kspon_jamo) | nfd_heard | 315.5M | 11.8% | 9.0% | 6/7 | 23/46 | mit | blank 연결 보정 |
| [speech31/XLS-R-300m-korean-ipa2](https://huggingface.co/speech31/XLS-R-300m-korean-ipa2) | ipa_explicit | 315.5M | 12.0% | 4.7% | 1/7 | 6/46 | 미기재 | — |
| [slplab/wav2vec2-xls-r-300m_phone-mfa_korean](https://huggingface.co/slplab/wav2vec2-xls-r-300m_phone-mfa_korean) | mfa | 315.5M | 12.4% | 9.5% | 4/7 | 13/46 | apache-2.0 | — |
| [Kkonjeong/wav2vec2-base-korean](https://huggingface.co/Kkonjeong/wav2vec2-base-korean) | compat_spelling | 94.4M | 14.1% | 5.3% | 2/7 | 9/46 | Apache 2.0 | — |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-s5](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-s5) | compat_spelling | 315.5M | 14.3% | 11.8% | 6/7 | 19/46 | apache-2.0 | — |
| [w11wo/wav2vec2-xls-r-300m-korean](https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean) | syllable | 316.7M | 14.3% | 3.8% | 1/7 | 0/46 | apache-2.0 | — |
| [tlsalfm820/wav2vec2-large-xls-r-korean](https://huggingface.co/tlsalfm820/wav2vec2-large-xls-r-korean) | compat_spelling | 315.5M | 14.4% | 9.7% | 2/7 | 10/46 | apache-2.0 | — |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | syllable | 966.4M | 14.9% | 13.0% | 0/7 | 0/46 | cc-by-nc-4.0 | — |
| [speech31/wavlm-large-korean-ipa](https://huggingface.co/speech31/wavlm-large-korean-ipa) | ipa_explicit | 315.5M | 14.9% | 7.6% | 5/7 | 8/46 | 미기재 | 실제 구조 Wav2Vec2ForCTC |
| [hyyoka/wav2vec2-xlsr-korean-senior](https://huggingface.co/hyyoka/wav2vec2-xlsr-korean-senior) | compat_spelling | 315.5M | 14.9% | 14.0% | 6/7 | 24/46 | apache-2.0 | blank 연결 보정 |
| [thisisHJLee/wav2vec2-large-xls-r-1b-korean-convsen1](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-1b-korean-convsen1) | compat_spelling | 962.6M | 15.6% | 13.4% | 5/7 | 12/46 | apache-2.0 | — |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-e](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-e) | compat_spelling | 315.5M | 15.8% | 12.3% | 3/7 | 10/46 | apache-2.0 | — |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-convsen2](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-convsen2) | compat_spelling | 315.5M | 16.2% | 15.1% | 6/7 | 14/46 | apache-2.0 | — |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-convsen3](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-convsen3) | compat_spelling | 315.5M | 16.3% | 14.6% | 6/7 | 18/46 | apache-2.0 | — |
| [thisisHJLee/wav2vec2-large-xls-r-1b-korean-convsen5](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-1b-korean-convsen5) | compat_spelling | 962.6M | 16.6% | 14.1% | 6/7 | 5/46 | apache-2.0 | — |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-s4](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-s4) | compat_spelling | 315.5M | 16.7% | 15.5% | 4/7 | 14/46 | apache-2.0 | — |
| [daeinbangeu/wav2vec2-large-xls-r-300m-korean-g](https://huggingface.co/daeinbangeu/wav2vec2-large-xls-r-300m-korean-g) | compat_spelling | 315.5M | 17.1% | 13.6% | 0/7 | 1/46 | 미기재 | 동일 가중치의 전처리 복원; 중복 가중치 |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-g](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-g) | compat_spelling | 315.5M | 17.1% | 13.6% | 0/7 | 1/46 | apache-2.0 | 중복 가중치 |
| [lsnoo/XLS-R-300m_35children_jamo](https://huggingface.co/lsnoo/XLS-R-300m_35children_jamo) | nfd_heard | 315.5M | 17.3% | 20.1% | 4/7 | 22/46 | mit | — |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-sen3](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-sen3) | compat_spelling | 315.5M | 17.5% | 15.8% | 6/7 | 14/46 | apache-2.0 | — |
| [icig/normal-korean-ipa-translation](https://huggingface.co/icig/normal-korean-ipa-translation) | icig_ipa | 94.4M | 18.5% | 13.3% | 2/7 | 4/46 | 미기재 | — |
| [kresnik/wav2vec2-large-xlsr-korean](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean) | syllable | 316.7M | 18.6% | 1.6% | 2/7 | 3/46 | apache-2.0 | — |
| [daeinbangeu/wav2vec2-large-xls-r-300m-korean-g-TW3](https://huggingface.co/daeinbangeu/wav2vec2-large-xls-r-300m-korean-g-TW3) | compat_spelling | 315.5M | 18.6% | 14.0% | 0/7 | 0/46 | apache-2.0 | — |
| [NX2411/wav2vec2-large-xlsr-korean-demo3](https://huggingface.co/NX2411/wav2vec2-large-xlsr-korean-demo3) | compat_spelling | 315.5M | 19.8% | 13.3% | 1/7 | 7/46 | apache-2.0 | — |
| [Hyuk/wav2vec2-korean-v2](https://huggingface.co/Hyuk/wav2vec2-korean-v2) | syllable | 316.9M | 20.7% | 2.1% | 0/7 | 5/46 | 미기재 | — |
| [daeinbangeu/wav2vec2-large-xls-r-300m-korean-g-TW3-backup](https://huggingface.co/daeinbangeu/wav2vec2-large-xls-r-300m-korean-g-TW3-backup) | compat_spelling | 315.5M | 21.0% | 15.5% | 0/7 | 0/46 | 미기재 | — |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-ws1](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-ws1) | compat_spelling | 315.5M | 21.3% | 18.7% | 4/7 | 26/46 | apache-2.0 | — |
| [fleek/wav2vec-large-xlsr-korean](https://huggingface.co/fleek/wav2vec-large-xlsr-korean) | compat_spelling | 315.5M | 21.4% | 19.7% | 5/7 | 15/46 | 미기재 | blank 연결 보정 |
| [slplab/wav2vec2-xlsr-korean-aihub-900](https://huggingface.co/slplab/wav2vec2-xlsr-korean-aihub-900) | mfa_characters | 315.5M | 21.8% | 20.4% | 2/7 | 8/46 | apache-2.0 | MFA 문자 조합 해석·민감도 별도 |
| [MaximTitarenkoUIT/phones_korean_from_53__4700](https://huggingface.co/MaximTitarenkoUIT/phones_korean_from_53__4700) | ipa_context | 315.5M | 22.4% | 28.4% | 0/7 | 0/46 | 미기재 | — |
| [MaximTitarenkoUIT/phones_korean_from_scratch__4800](https://huggingface.co/MaximTitarenkoUIT/phones_korean_from_scratch__4800) | ipa_context | 315.5M | 22.5% | 27.0% | 2/7 | 16/46 | 미기재 | — |
| [icig/announcer-korean-ipa-translation](https://huggingface.co/icig/announcer-korean-ipa-translation) | icig_ipa | 94.4M | 25.4% | 20.1% | 0/7 | 9/46 | 미기재 | — |
| [NX2411/wav2vec2-large-xlsr-korean-demo-with-LM](https://huggingface.co/NX2411/wav2vec2-large-xlsr-korean-demo-with-LM) | compat_spelling | 315.5M | 26.3% | 25.0% | 1/7 | 13/46 | apache-2.0 | — |
| [NX2411/wav2vec2-large-xlsr-korean-demo-no-LM](https://huggingface.co/NX2411/wav2vec2-large-xlsr-korean-demo-no-LM) | compat_spelling | 315.5M | 27.4% | 25.8% | 2/7 | 10/46 | apache-2.0 | — |
| [NX2411/wav2vec2-large-xlsr-korean-demo-colab-2](https://huggingface.co/NX2411/wav2vec2-large-xlsr-korean-demo-colab-2) | compat_spelling | 315.5M | 27.4% | 28.4% | 3/7 | 8/46 | apache-2.0 | — |
| [NX2411/wav2vec2-large-xlsr-korean-demo-test](https://huggingface.co/NX2411/wav2vec2-large-xlsr-korean-demo-test) | compat_spelling | 315.6M | 27.4% | 26.1% | 6/7 | 15/46 | apache-2.0 | — |
| [anantoj/wav2vec2-xls-r-1b-korean](https://huggingface.co/anantoj/wav2vec2-xls-r-1b-korean) | syllable | 964.0M | 27.5% | 1.2% | 0/7 | 0/46 | apache-2.0 | — |
| [lsnoo/xlsr-53_aihub_35children_jamo](https://huggingface.co/lsnoo/xlsr-53_aihub_35children_jamo) | nfd_heard | 315.5M | 29.9% | 41.6% | 3/7 | 17/46 | mit | blank 연결 보정 |
| [slplab/wav2vec2-xls-r-300m_phoneme-mfa_korean_nia13-asia-9634_001](https://huggingface.co/slplab/wav2vec2-xls-r-300m_phoneme-mfa_korean_nia13-asia-9634_001) | mfa | 315.5M | 31.5% | 29.9% | 4/7 | 11/46 | apache-2.0 | — |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-w1](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-w1) | compat_spelling | 315.5M | 33.9% | 33.7% | 6/7 | 22/46 | apache-2.0 | — |
| [raniczan/wav2vec2-xls-r-300m-korean-english](https://huggingface.co/raniczan/wav2vec2-xls-r-300m-korean-english) | nfd_spelling | 315.9M | 35.1% | 35.7% | 2/7 | 12/46 | apache-2.0 | — |
| [SudheenK/xlsr300m-korean-ctc](https://huggingface.co/SudheenK/xlsr300m-korean-ctc) | compat_spelling | 315.5M | 50.7% | 32.0% | 0/7 | 8/46 | apache-2.0 | — |
| [slplab/wav2vec2-XLSR-300m_KoreanPhonene_spoken_by_foreigners](https://huggingface.co/slplab/wav2vec2-XLSR-300m_KoreanPhonene_spoken_by_foreigners) | foreign_heard | 315.5M | 55.3% | 50.8% | 0/7 | 1/46 | 미기재 | — |
| [slplab/wav2vec2_xlsr50k_korean_phoneme_aihub-40m](https://huggingface.co/slplab/wav2vec2_xlsr50k_korean_phoneme_aihub-40m) | mfa | 315.5M | 71.7% | 71.2% | 0/7 | 0/46 | apache-2.0 | — |
| [jaeyeon/korean-aihub-learning-3](https://huggingface.co/jaeyeon/korean-aihub-learning-3) | syllable | 316.0M | 72.3% | 66.9% | 0/7 | 0/46 | apache-2.0 | — |
| [N03N9/wav2vec2-korean-ipa](https://huggingface.co/N03N9/wav2vec2-korean-ipa) | ipa_context | 315.5M | 87.0% | 85.4% | 0/7 | 0/46 | 미기재 | — |
| [rosicast/1205-wav2vec2-large-xlsr-53-korean-zeroth-char](https://huggingface.co/rosicast/1205-wav2vec2-large-xlsr-53-korean-zeroth-char) | syllable | 326.9M | 91.4% | 89.4% | 0/7 | 0/46 | 미기재 | — |
| [ppparkker/wav2vec2-xls-r-300m-S2P-korean](https://huggingface.co/ppparkker/wav2vec2-xls-r-300m-S2P-korean) | compat_heard | 315.5M | 92.8% | 94.4% | 0/7 | 0/46 | apache-2.0 | — |
| [tlsalfm820/wav2vec2-large-xls-r-300m-korean-g](https://huggingface.co/tlsalfm820/wav2vec2-large-xls-r-300m-korean-g) | compat_spelling | 315.5M | 93.8% | 93.9% | 0/7 | 0/46 | apache-2.0 | — |
| [eunyounglee/wav2vec_korean](https://huggingface.co/eunyounglee/wav2vec_korean) | compat_spelling | 315.5M | 98.5% | 99.0% | 0/7 | 0/46 | apache-2.0 | — |
| [chaos6749/wav2vec2-large-xlrs-korean-v3](https://huggingface.co/chaos6749/wav2vec2-large-xlrs-korean-v3) | compat_spelling | 315.5M | 100.0% | 100.0% | 0/7 | 0/46 | apache-2.0 | — |
| [dgkim0306/wav2vec2-large-xls-r-300m-korean-g](https://huggingface.co/dgkim0306/wav2vec2-large-xls-r-300m-korean-g) | compat_spelling | 315.5M | 100.0% | 100.0% | 0/7 | 0/46 | apache-2.0 | — |
| [thisisHJLee/wav2vec2-large-xls-r-300m-korean-b](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-b) | compat_spelling | 315.5M | 141.7% | 113.4% | 0/7 | 0/46 | apache-2.0 | — |
| [slplab/wav2vec2-large-xlsr-53-korean-nia13-asia-9634_001](https://huggingface.co/slplab/wav2vec2-large-xlsr-53-korean-nia13-asia-9634_001) | mfa_characters | 315.5M | 176.1% | 172.5% | 0/7 | 0/46 | apache-2.0 | MFA 문자 조합 해석·민감도 별도 |

**실제 추론했으나 소리 오류율 순위에서 제외**

`miranoh/wav2vec2-xlsr-53-korean`: 전 입력 추론 완료. 공개 출력 기호와 발음의 대응을 확인할 수 없어 임의 대응으로 점수를 만들지 않았다.

`thisisHJLee/wav2vec2-large-xls-r-1b-korean-sample5`: 출력층 109칸·모델 blank 108과 배포 vocab 93개·tokenizer pad 92가 불일치한다. 공개 이력 66개를 검사해도 전체 매핑이 없었다. 모델 지정 blank를 사용해 3,157개 입력의 CTC 번호열을 보존했으며, 모르는 출력 번호에 글자를 임의로 붙여 점수를 만들지 않았다.

**실제 추론하지 않은 저장소의 조사 상태**

음절 대조군은 대표 후보를 실행했다. 나머지 모든 음절 ASR·모든 구조를 전수 비교했다는 뜻은 아니다. 분류·인코더 기반 모델을 인식 성능이 낮다고 탈락시킨 것이 아니다. 공개 파일이 불완전하거나 접근할 수 없는 후보도 성능 실패와 구분했다.

| 조사 상태 | 저장소 수 |
|---|---|
| excluded_audio_classification | 43 |
| metadata_unavailable | 2 |
| missing_output_mapping_or_weights | 1 |
| missing_root_weights_or_output_mapping | 28 |
| no_ready_ctc_head | 6 |
| syllable_or_other_output_not_selected_for_jamo_phone_sweep | 46 |
| tested_raw_inference_only | 2 |
| tested_scored | 62 |
| unverified_english_alphabet_output | 3 |

| 저장소 | 상태 |
|---|---|
| Arering/korean_kws | excluded_audio_classification |
| Flitto/wav2vec2-base-korean-dialect-recognition | missing_root_weights_or_output_mapping |
| Frost0098/korean_kws2 | excluded_audio_classification |
| Gummybear05/wav2vec2-large-xls-r-300m-korean_senior-colab | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| Gummybear05/wav2vec2-large-xls-r-300m-korean_speed | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| HERIUN/wav2vec2-xlsr-korean-dialect-recognition | missing_root_weights_or_output_mapping |
| Hyuk/wav2vec2-korean-v1 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| Hyuk/wav2vec2-korean-v3 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| JungYongjun/wav2vec2-korean-asr | missing_root_weights_or_output_mapping |
| Jungwonchang/wav2vec2-large-mms-1b-korean-colab | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| KimChanYu/wav2vec2-korean-wav-multi-emotion | no_ready_ctc_head |
| Lucas37/korean_kws | excluded_audio_classification |
| Lypew/korean_kws2 | excluded_audio_classification |
| Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| NX2411/wav2vec2-large-xlsr-korean-demo-test2 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| Reggis/wav2vec2-xls-r-300m_phone-mfa_korean | no_ready_ctc_head |
| Seyeon1027/korean_kws | excluded_audio_classification |
| Wing4/base_korean_audio_cls | excluded_audio_classification |
| ZZori/wav2vec2-korean-intent | excluded_audio_classification |
| ZZori/wav2vec2-korean-stt-minds14 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| allzero0102/korean_kws2 | excluded_audio_classification |
| alwaysnotavailable/korean_kws | excluded_audio_classification |
| bsb0107/korean_kws | excluded_audio_classification |
| byc3230/hubert-large-korean-finetuned-korspeech-ser | excluded_audio_classification |
| byc3230/hubert-large-korean-finetuned-korspeech-ser2 | excluded_audio_classification |
| byhylee/korean_kws2 | excluded_audio_classification |
| cheolminBae/korean_kws | excluded_audio_classification |
| daeinbangeu/wav2vec2-large-xls-r-300m-korean-TW3 | missing_root_weights_or_output_mapping |
| daeinbangeu/wav2vec2-large-xls-r-300m-korean-g-tw1 | missing_root_weights_or_output_mapping |
| daeunn/wav2vec2-finetuned-korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| daeunn/wav2vec2-korean-finetuned2 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| daeunn/wav2vec2-korean-finetuned2-loss | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| devtaco/korean_kws2 | excluded_audio_classification |
| diegolee5494/asr-korean-mind-model | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| djifg/wav2vec2-korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| dwdw-song/wav2vec2-large-xlsr-korean | missing_root_weights_or_output_mapping |
| ella0106/wav2vec2-large-xlsr-korean-4 | missing_root_weights_or_output_mapping |
| eunyounglee/wav2vec2-large-xls-r-300m-korean-third | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| furence-ai/korean-ser-wav2vec2-xlsr-4class | excluded_audio_classification |
| gharam12/korean_kws2 | excluded_audio_classification |
| hahah121212/korean_kws2 | excluded_audio_classification |
| hansol23/korean_kws2 | excluded_audio_classification |
| heado/korean_kws | excluded_audio_classification |
| heado/korean_kws2 | excluded_audio_classification |
| huni0304/XLS-R-300m-korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| hwirang/korean_kws2 | excluded_audio_classification |
| hyeoncheon/korean_kws2 | excluded_audio_classification |
| hyojin99/wav2vec2-large-xls-r-300m-korean-g | missing_root_weights_or_output_mapping |
| hyorea1/wav2vec2-korean-test | missing_root_weights_or_output_mapping |
| ijin07/wav2vec2-large-xlsr-53-korean | missing_root_weights_or_output_mapping |
| inseong00/wav2vec2-large-xlsr-korean-autumn | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| j911214/korean_kws2 | excluded_audio_classification |
| jaeyeon/korean-aihub-learning-2 | missing_root_weights_or_output_mapping |
| jaeyeon/korean-aihub-learning-math-1-test | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| jaeyeon/korean-aihub-learning-math-16batch | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| jaeyeon/korean-aihub-learning-math-8batch | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| jaeyeon/korean-aihub-learning-math-H02-p3-1gpu-220928 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| jandyjeong/korean_kws2 | excluded_audio_classification |
| jandyjeong/korean_kws3 | excluded_audio_classification |
| jonatasgrosman/wav2vec2-large-xlsr-53-korean | metadata_unavailable |
| julieseo0705/asr-korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| jungjongho/wav2vec2-large-xlsr-korean-demo-colab | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| jungjongho/wav2vec2-large-xlsr-korean-demo-colab_epoch15 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition | no_ready_ctc_head |
| jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance | no_ready_ctc_head |
| jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition3 | no_ready_ctc_head |
| kimsoob/korean_kws | excluded_audio_classification |
| kresnik/wav2vec2-large-xls-r-1b-korean | metadata_unavailable |
| lightwsrld/wav2vec2-large-xlsr-korean-autumn | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| lsnoo/xlsr-53-korean_kids_3_to_5_syl | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| mhahyeon/korean_kws2 | excluded_audio_classification |
| miranoh/wav2vec2-large-xlsr-53-korean | missing_root_weights_or_output_mapping |
| molto/ft_0112_korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| namvandy/wav2vec2-base-zeroth_korean-demo | missing_root_weights_or_output_mapping |
| namvandy/wav2vec2-large-zeroth_korean-demo-221213_2 | missing_root_weights_or_output_mapping |
| namvandy/wav2vec2-large-zeroth_korean_221213 | missing_root_weights_or_output_mapping |
| nunuing/audio_cls_korean | excluded_audio_classification |
| nunuing/korean_kws2 | missing_root_weights_or_output_mapping |
| okjoo1/korean_kws2 | excluded_audio_classification |
| pongjin/en_with_korean_model_large_960h | unverified_english_alphabet_output |
| pongjin/en_with_korean_w2v_model | unverified_english_alphabet_output |
| pongjin/en_with_korean_w2v_model_960h | unverified_english_alphabet_output |
| puppyAhn/korean_kws | excluded_audio_classification |
| rewining/korean_kws2 | excluded_audio_classification |
| rkdwo/korean_kws2 | excluded_audio_classification |
| rosicast/1205-wav2vec2-large-xlsr-53-korean-zeroth-char_ulsan | missing_root_weights_or_output_mapping |
| ryulee2ppi/korean_kws2 | excluded_audio_classification |
| seny1004/wav2vec2-large-mms-1b-korean-colab | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| seny1004/wav2vec2-large-mms-korean-colab | missing_root_weights_or_output_mapping |
| seohyeon122333/korean_kws2 | excluded_audio_classification |
| slplab/wav2vec2-large-robust_ETRI_Korean_english-pronunciation | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| slplab/wav2vec2-xls-r_Korean_ASR_by_foreigners | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| slplab/wav2vec_xlsr50k_korean_phoneme_test | missing_output_mapping_or_weights |
| soybean1102/korean_kws2 | excluded_audio_classification |
| student-47/wav2vec2-large-xlrs-korean-v6 | missing_root_weights_or_output_mapping |
| sudopop/wav2vec2-xls-r-300m-korean | missing_root_weights_or_output_mapping |
| sungkwan2/korean_kws2 | excluded_audio_classification |
| sungkwan2/minds14_wav2vec2-korean | excluded_audio_classification |
| syp1229/wav2vec2-large-xls-r-300m-korean-d | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| syp1229/wav2vec2-large-xls-r-300m-korean_y_freq | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| team-lucid/hubert-base-korean | no_ready_ctc_head |
| team-lucid/hubert-large-korean | missing_root_weights_or_output_mapping |
| team-lucid/hubert-xlarge-korean | missing_root_weights_or_output_mapping |
| teddy322/wav2vec2-large-xls-r-300m-korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| teems79/korean_kws2 | excluded_audio_classification |
| teems79/korean_kws2-1 | excluded_audio_classification |
| teems79/wav2vec2-base-korean | excluded_audio_classification |
| thisisHJLee/wav2vec2-large-xls-r-1b-korean-convsen2 | missing_root_weights_or_output_mapping |
| thisisHJLee/wav2vec2-large-xls-r-300m-korean-convsen | missing_root_weights_or_output_mapping |
| thisisHJLee/wav2vec2-large-xls-r-300m-korean-f | missing_root_weights_or_output_mapping |
| thisisHJLee/wav2vec2-large-xls-r-300m-korean-g-tw1 | missing_root_weights_or_output_mapping |
| thisisHJLee/wav2vec2-large-xls-r-300m-korean-s2 | missing_root_weights_or_output_mapping |
| w11wo/wav2vec2-xls-r-300m-korean-lm | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| weekcircle/wav2vec2-large-mms-1b-korean-colab | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| weekcircle/wav2vec2-large-mms-1b-korean-colab_v0 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| weekcircle/wav2vec2-large-mms-1b-korean-colab_v2 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| weekcircle/wav2vec2-large-mms-1b-korean-colab_v3 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| xonme888/wav2vec2-korean-finetuned | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| y981010/anantoj-wav2vec2-xls-r-1b-korean-colab | missing_root_weights_or_output_mapping |
| yaongua/korean_kws2 | excluded_audio_classification |
| yoon1000/Korean_STT_v0 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| yoon1000/ft_0117_korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| yoon1000/ft_0122_korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| yoon1000/ft_0123_korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| yoon1000/ft_0123_korean_1 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| yoon1000/ft_0124_korean_1 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| yoon1000/ft_0124_korean_2 | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| yoon1000/ft_0202_korean | syllable_or_other_output_not_selected_for_jamo_phone_sweep |
| yunye99/korean_kws2 | excluded_audio_classification |

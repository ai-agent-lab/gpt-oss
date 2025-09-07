# Is GPT-OSS Good? A Comprehensive Evaluation of OpenAI's Latest Open Source Models

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2508.12461)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/ai-agent-lab/gpt-oss)
[![Website](https://img.shields.io/badge/Website-Project%20Page-blue)](https://github.com/ai-agent-lab/gpt-oss/)

**Authors:** Ziqian Bi¹'²*, Keyu Chen¹'³*, Chiung-Yi Tseng¹'⁴*, Danyang Zhang¹'⁵*, Tianyang Wang¹, Hongying Luo¹, Lu Chen¹, Junming Huang¹, Jibin Guan⁶, Junfeng Hao⁶, Junhao Song⁷†

<sup>*</sup>Indicates Equal Contribution, <sup>†</sup>Corresponding Author

**Affiliations:**
- ¹ AI Agent Lab, Vokram Group, UK
- ² Purdue University, USA
- ³ Georgia Institute of Technology, USA
- ⁴ LuxMuse AI, USA
- ⁵ ByteDance Inc, USA
- ⁶ University of Minnesota, USA
- ⁷ Imperial College London, UK

## Abstract

In August 2025, OpenAI released GPT-OSS models, its first open weight large language models since GPT-2 in 2019, comprising two mixture of experts architectures with 120B and 20B parameters. We evaluated both variants against six contemporary open source large language models ranging from 14.7B to 235B parameters, representing both dense and sparse designs, across ten benchmarks covering general knowledge, mathematical reasoning, code generation, multilingual understanding, and conversational ability. All models were tested in unquantised form under standardised inference settings, with statistical validation using McNemar's test and effect size analysis. Results show that gpt-oss-20B consistently outperforms gpt-oss-120B on several benchmarks, such as HumanEval and MMLU, despite requiring substantially less memory and energy per response. Both models demonstrate mid-tier overall performance within the current open source landscape, with relative strength in code generation and notable weaknesses in multilingual tasks. These findings provide empirical evidence that scaling in sparse architectures may not yield proportional performance gains, underscoring the need for further investigation into optimisation strategies and informing more efficient model selection for future open source deployments.

## Research Figures

### Figure 1: Performance Rankings
![Performance Rankings](static/images/img/figure_1_overall_ranking.png)
*Performance rankings across benchmark categories using general prompts. Error bars represent 95% confidence intervals.*

### Figure 2: Performance Heatmap
![Performance Heatmap](static/images/img/figure_2_performance_heatmap.png)
*Performance heatmap across model-benchmark combinations. Darker blue indicates higher accuracy.*

### Figure 3: Task Categories
![Task Categories](static/images/img/figure_3_task_categories.png)
*Performance distribution across evaluation categories. Analysis methodology follows BIG-bench protocols.*

### Figure 4: Parameter-Performance Relationship
![Parameter-Performance](static/images/img/figure_4_model_size_comparison.png)
*Parameter-performance relationship. The non-monotonic scaling observed in GPT-OSS variants contradicts established scaling laws.*

### Figure 5: GPT-OSS Detailed Comparison
![GPT-OSS Detailed](static/images/img/figure_5_gpt_oss_detailed.png)
*Direct performance comparison between GPT-OSS variants across all evaluation benchmarks.*

### Figure 6: Multi-dimensional Performance
![Multi-dimensional Performance](static/images/img/figure_6_radar_chart.png)
*Multi-dimensional performance comparison across eight evaluated models. GPT-OSS models show middle-tier performance.*

### Figure 7: Token Count Distribution
![Token Distribution](static/images/img/figure_7_token_histograms.png)
*Token count distribution across all models on aggregated datasets. GPT-OSS models exhibit notably concise outputs.*

## Key Findings

- **GPT-OSS-20B outperforms GPT-OSS-120B** on several benchmarks including HumanEval and MMLU
- **Non-monotonic scaling** in sparse architectures challenges established scaling laws
- **Mid-tier performance** within the current open source landscape
- **Strong code generation** capabilities but weaknesses in multilingual tasks
- **Energy efficiency** advantages of the smaller 20B model

## Citation

```bibtex
@misc{bi2025gptossgoodcomprehensiveevaluation,
  title={Is GPT-OSS Good? A Comprehensive Evaluation of OpenAI's Latest Open Source Models}, 
  author={Ziqian Bi and Keyu Chen and Chiung-Yi Tseng and Danyang Zhang and Tianyang Wang and Hongying Luo and Lu Chen and Junming Huang and Jibin Guan and Junfeng Hao and Junhao Song},
  year={2025},
  eprint={2508.12461},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2508.12461}, 
}
```

## References & Related Work

### LLM Surveys & Reviews
- *A survey of large language models*, Zhao et al. [[arXiv 2023]](https://arxiv.org/abs/2303.18223)
- *Large language models: A survey*, Minaee et al. [[arXiv 2024]](https://arxiv.org/abs/2402.06196)
- *Eight things to know about large language models*, Bowman [[arXiv 2023]](https://arxiv.org/abs/2304.00612)
- *Unifying the perspectives of NLP and software engineering: A survey on language models for code*, Zhang et al. [[arXiv 2023]](https://arxiv.org/abs/2311.07989)

### Foundation Models & Scaling Laws
- *Language models are few-shot learners*, Brown et al. [[arXiv 2020]](https://arxiv.org/abs/2006.14701)
- *Language models are unsupervised multitask learners*, Radford et al. [[OpenAI 2019]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- *Scaling laws for neural language models*, Kaplan et al. [[arXiv 2020]](https://arxiv.org/abs/2001.08361)
- *Training compute-optimal large language models*, Hoffmann et al. [[arXiv 2022]](https://arxiv.org/abs/2203.15556)
- *Scaling laws for autoregressive generative modeling*, Henighan et al. [[arXiv 2020]](https://arxiv.org/abs/2010.14701)
- *Emergent abilities of large language models*, Wei et al. [[arXiv 2022]](https://arxiv.org/abs/2206.07682)
- *Are emergent abilities of large language models a mirage?*, Schaeffer et al. [[NeurIPS 2024]](https://proceedings.neurips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf)

### Model Architectures & Mixture of Experts
- *Attention is all you need*, Vaswani et al. [[NeurIPS 2017]](https://arxiv.org/abs/1706.03762)
- *Outrageously large neural networks: The sparsely-gated mixture-of-experts layer*, Shazeer et al. [[ICLR 2017]](https://arxiv.org/abs/1701.06538)
- *Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity*, Fedus et al. [[JMLR 2022]](https://arxiv.org/abs/2101.03961)
- *Glam: Efficient scaling of language models with mixture-of-experts*, Du et al. [[ICML 2022]](https://arxiv.org/abs/2112.06905)
- *Gshard: Scaling giant models with conditional computation and automatic sharding*, Lepikhin et al. [[arXiv 2021]](https://arxiv.org/abs/2006.16668)
- *Mixture-of-experts with expert choice routing*, Zhou et al. [[NeurIPS 2022]](https://arxiv.org/abs/2202.09368)
- *Lamda: Language models for dialog applications*, Thoppilan et al. [[arXiv 2022]](https://arxiv.org/abs/2201.08239)
- *Scaling language models: Methods, analysis & insights from training gopher*, Rae et al. [[arXiv 2021]](https://arxiv.org/abs/2112.11446)
- *Palm: Scaling language modeling with pathways*, Chowdhery et al. [[arXiv 2022]](https://arxiv.org/abs/2204.02311)
- *Efficient large scale language modeling with mixtures of experts*, Artetxe et al. [[arXiv 2021]](https://arxiv.org/abs/2112.10684)
- *mT5: A massively multilingual pre-trained text-to-text transformer*, Xue et al. [[arXiv 2021]](https://arxiv.org/abs/2010.11934)
- *Dialogpt: Large-scale generative pre-training for conversational response generation*, Zhang et al. [[arXiv 2020]](https://arxiv.org/abs/1911.00536)
- *Lifting the curse of multilinguality by pre-training modular transformers*, Pfeiffer et al. [[arXiv 2022]](https://arxiv.org/abs/2205.06266)
- *A neural network solves, explains, and generates university math problems by program synthesis and few-shot learning at human level*, Drori et al. [[PNAS 2022]](https://arxiv.org/abs/2112.15594)
- *The goldilocks of pragmatic understanding: Fine-tuning strategy matters for implicature resolution by llms*, Ruis et al. [[NeurIPS 2024]](https://arxiv.org/abs/2210.14986)
- *llama.cpp: Inference of LLaMA model in pure C/C++*, Gerganov [[GitHub 2023]](https://github.com/ggerganov/llama.cpp)
- *Parallel machine translation with disentangled context transformer*, Kasai et al. [[ICML 2021]](https://arxiv.org/abs/2001.05136)
- *Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale*, Dettmers et al. [[NeurIPS 2022]](https://arxiv.org/abs/2208.07339)
- *Retrieval-augmented generation for knowledge-intensive nlp tasks*, Lewis et al. [[NeurIPS 2020]](https://arxiv.org/abs/2005.11401)
- *Prior and Posterior Networks: A Survey on Evidential Deep Learning Methods For Uncertainty Estimation*, Ulmer et al. [[arXiv 2022]](https://arxiv.org/abs/2110.03051)
- *Claude Opus 4.1*, Anthropic [[Company 2025]](https://www.anthropic.com/news/claude-opus-4-1)

### Contemporary Open Source Models
- *Llama: Open and efficient foundation language models*, Touvron et al. [[arXiv 2023]](https://arxiv.org/abs/2302.13971)
- *Llama 2: Open foundation and fine-tuned chat models*, Touvron et al. [[arXiv 2023]](https://arxiv.org/abs/2307.09288)
- *Gemma: Open models based on gemini research and technology*, Gemma Team [[arXiv 2024]](https://arxiv.org/abs/2403.08295)
- *Gemini: a family of highly capable multimodal models*, Team et al. [[arXiv 2023]](https://arxiv.org/abs/2312.11805)
- *Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities*, Comanici et al. [[arXiv 2025]](https://arxiv.org/abs/2507.06261)
- *DeepSeek LLM: Scaling open-source language models with longtermism*, DeepSeek-AI [[arXiv 2024]](https://arxiv.org/abs/2401.02954)
- *Qwen technical report*, Bai et al. [[arXiv 2023]](https://arxiv.org/abs/2309.16609)
- *Qwen2.5 technical report*, Qwen Team [[arXiv 2024]](https://arxiv.org/abs/2409.12186)
- *Qwen3 Technical Report*, Yang et al. [[arXiv 2025]](https://arxiv.org/abs/2505.09388)
- *Phi-3 technical report: A highly capable language model locally on your phone*, Abdin et al. [[arXiv 2024]](https://arxiv.org/abs/2404.14219)
- *The falcon series of open language models*, Almazrouei et al. [[arXiv 2023]](https://arxiv.org/abs/2311.16867)
- *Mistral 7B*, Jiang et al. [[arXiv 2023]](https://arxiv.org/abs/2310.06825)

### Evaluation & Benchmarking
- *Measuring massive multitask language understanding*, Hendrycks et al. [[arXiv 2021]](https://arxiv.org/abs/2009.03300)
- *Holistic evaluation of language models*, Liang et al. [[TMLR 2023]](https://arxiv.org/abs/2211.09110)
- *Beyond the imitation game: Quantifying and extrapolating the capabilities of language models*, Srivastava et al. [[arXiv 2022]](https://arxiv.org/abs/2206.04615)
- *Judging LLM-as-a-judge with MT-bench and chatbot arena*, Zheng et al. [[arXiv 2023]](https://arxiv.org/abs/2306.05685)
- *Training verifiers to solve math word problems*, Cobbe et al. [[arXiv 2021]](https://arxiv.org/abs/2110.14168)
- *Evaluating large language models trained on code*, Chen et al. [[arXiv 2021]](https://arxiv.org/abs/2107.03374)
- *C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models*, Huang et al. [[NeurIPS 2023]](https://arxiv.org/abs/2305.08322)

### Domain-Specific Benchmarks
- *FinQA: A dataset of numerical reasoning over financial data*, Chen et al. [[arXiv 2021]](https://arxiv.org/abs/2109.00122)
- *What disease does this patient have? a large-scale open domain question answering dataset from medical exams*, Jin et al. [[Applied Sciences 2021]](https://www.mdpi.com/2076-3417/11/14/6421)
- *Experimenting with legal ai solutions: The case of question-answering for access to justice*, Li et al. [[arXiv 2024]](https://arxiv.org/abs/2409.07713)
- *Crowdsourcing multiple choice science questions*, Welbl et al. [[arXiv 2017]](https://arxiv.org/abs/1707.06209)
- *Piqa: Reasoning about physical commonsense in natural language*, Bisk et al. [[AAAI 2020]](https://arxiv.org/abs/1911.11641)
- *DialogSum: A real-life scenario dialogue summarization dataset*, Chen et al. [[arXiv 2021]](https://arxiv.org/abs/2105.06762)
- *Benchmarking the performance of large language models in uveitis: a comparative analysis of ChatGPT-3.5, ChatGPT-4.0, Google Gemini, and Anthropic Claude3*, Zhao et al. [[arXiv 2025]](https://arxiv.org/abs/2501.01234)
- *Large language models are not fair evaluators*, Wang et al. [[arXiv 2023]](https://arxiv.org/abs/2305.17926)
- *AI and the everything in the whole wide world benchmark*, Raji et al. [[NeurIPS 2021]](https://arxiv.org/abs/2111.15366)
- *Beyond accuracy: Behavioral testing of NLP models with CheckList*, Ribeiro et al. [[ACL 2020]](https://aclanthology.org/2020.acl-main.442/)
- *The GEM benchmark: Natural language generation, its evaluation and metrics*, Gehrmann et al. [[arXiv 2021]](https://arxiv.org/abs/2102.01672)
- *Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation*, Hu et al. [[arXiv 2020]](https://arxiv.org/abs/2003.11080)
- *CRUXEval: A benchmark for code reasoning, understanding and execution*, Gu et al. [[arXiv 2024]](https://arxiv.org/abs/2401.03065)
- *Medical Exam Question Answering with Large-scale Reading Comprehension*, Zhang et al. [[arXiv 2018]](https://arxiv.org/abs/1809.09687)
- *Measuring and improving compositional generalization in text-to-sql via component alignment*, Gan et al. [[NAACL 2022]](https://aclanthology.org/2022.naacl-main.281/)
- *The benchmark lottery*, Dehghani et al. [[arXiv 2021]](https://arxiv.org/abs/2107.07002)
- *Rethink reporting of evaluation results in AI*, Burnell et al. [[arXiv 2023]](https://arxiv.org/abs/2211.08571)
- *NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark*, Sainz et al. [[arXiv 2023]](https://arxiv.org/abs/2310.18018)

### Reasoning & Code Generation
- *Chain-of-thought prompting elicits reasoning in large language models*, Wei et al. [[NeurIPS 2022]](https://arxiv.org/abs/2201.11903)
- *Large language models are zero-shot reasoners*, Kojima et al. [[NeurIPS 2022]](https://arxiv.org/abs/2205.11916)
- *Self-consistency improves chain of thought reasoning in language models*, Wang et al. [[arXiv 2023]](https://arxiv.org/abs/2203.11171)
- *Program synthesis with large language models*, Austin et al. [[arXiv 2021]](https://arxiv.org/abs/2108.07732)
- *StarCoder: may the source be with you!*, Li et al. [[arXiv 2023]](https://arxiv.org/abs/2305.06161)
- *Codegen: An open large language model for code with multi-turn program synthesis*, Nijkamp et al. [[arXiv 2022]](https://arxiv.org/abs/2203.13474)
- *Incoder: A generative model for code infilling and synthesis*, Fried et al. [[arXiv 2022]](https://arxiv.org/abs/2204.05999)
- *Solving quantitative reasoning problems with language models*, Lewkowycz et al. [[NeurIPS 2022]](https://arxiv.org/abs/2206.14858)
- *Impact of pretraining term frequencies on few-shot numerical reasoning*, Razeghi et al. [[arXiv 2022]](https://arxiv.org/abs/2202.07206)
- *Are NLP models really able to solve simple math word problems?*, Patel et al. [[arXiv 2021]](https://arxiv.org/abs/2103.07191)
- *Inverse scaling can become U-shaped*, Wei et al. [[arXiv 2023]](https://arxiv.org/abs/2211.02011)
- *Inverse scaling: When bigger isn't better*, McKenzie et al. [[arXiv 2023]](https://arxiv.org/abs/2306.09479)

### Statistical Analysis & Methodology
- *The hitchhiker's guide to testing statistical significance in natural language processing*, Dror et al. [[ACL 2018]](https://aclanthology.org/P18-1128/)
- *Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis*, Benavoli et al. [[JMLR 2017]](https://jmlr.org/papers/v18/16-305.html)
- *With Little Power Comes Great Responsibility*, Card et al. [[EMNLP 2020]](https://aclanthology.org/2020.emnlp-main.745/)
- *An introduction to the bootstrap*, Efron & Tibshirani [Book 1994]
- *Statistical power analysis for the behavioral sciences*, Cohen [Book 1988]
- *A statistical analysis of summarization evaluation metrics using resampling methods*, Deutsch et al. [[NAACL 2021]](https://aclanthology.org/2021.naacl-main.287/)
- *Statistical significance tests for machine translation evaluation*, Koehn [[WMT 2004]](https://aclanthology.org/W04-3250/)
- *Controlling the false discovery rate: a practical and powerful approach to multiple testing*, Benjamini & Hochberg [[JRSS-B 1995]](https://www.jstor.org/stable/2346101)
- *Improving reproducibility in machine learning research: a report from the NeurIPS 2019 reproducibility program*, Pineau et al. [[JMLR 2021]](https://jmlr.org/papers/v22/20-303.html)
- *Show your work: Improved reporting of experimental results*, Dodge et al. [[EMNLP 2019]](https://aclanthology.org/D19-1224/)
- *Reporting score distributions makes a difference: Performance study of LSTM-networks for sequence tagging*, Reimers & Gurevych [[EMNLP 2017]](https://aclanthology.org/D17-1035/)
- *New effect size rules of thumb*, Sawilowsky [[JEP 2009]](https://www.tandfonline.com/doi/abs/10.1080/00220970903292900)
- *The measurement of observer agreement for categorical data*, Landis & Koch [[Biometrics 1977]](https://www.jstor.org/stable/2529310)
- *Note on the sampling error of the difference between correlated proportions or percentages*, McNemar [[Psychometrika 1947]](https://link.springer.com/article/10.1007/BF02295996)
- *Teoria statistica delle classi e calcolo delle probabilita*, Bonferroni [Book 1936]
- *Qa dataset explosion: A taxonomy of nlp resources for question answering and reading comprehension*, Rogers et al. [[arXiv 2023]](https://arxiv.org/abs/2301.05020)

### Efficiency & Environmental Impact
- *Energy and policy considerations for deep learning in NLP*, Strubell et al. [[ACL 2019]](https://aclanthology.org/P19-1355/)
- *Carbon emissions and large neural network training*, Patterson et al. [[arXiv 2021]](https://arxiv.org/abs/2104.10350)
- *Towards the systematic reporting of the energy and carbon footprints of machine learning*, Henderson et al. [[JMLR 2020]](https://jmlr.org/papers/v21/20-312.html)
- *Efficiently scaling transformer inference*, Pope et al. [[MLSys 2023]](https://proceedings.mlsys.org/paper_files/paper/2023/file/523f87e9d08e6071a3bbd150e6da40fb-Paper-mlsys2023.pdf)
- *Scale efficiently: Insights from pretraining and finetuning transformers*, Tay et al. [[arXiv 2022]](https://arxiv.org/abs/2109.10686)
- *Green ai*, Schwartz et al. [[CACM 2020]](https://cacm.acm.org/magazines/2020/12/248800-green-ai/fulltext)

### Multilingual & Cross-lingual Models
- *Language models are few-shot multilingual learners*, Winata et al. [[NAACL 2021]](https://aclanthology.org/2021.naacl-main.410/)
- *Unsupervised cross-lingual representation learning at scale*, Conneau et al. [[ACL 2020]](https://aclanthology.org/2020.acl-main.747/)
- *Crosslingual generalization through multitask finetuning*, Muennighoff et al. [[arXiv 2023]](https://arxiv.org/abs/2211.01786)

### Dialogue & Conversational AI
- *Recipes for building an open-domain chatbot*, Roller et al. [[EACL 2021]](https://aclanthology.org/2021.eacl-main.24/)
- *What makes a good conversation? How controllable attributes affect human judgments*, See et al. [[NAACL 2019]](https://aclanthology.org/N19-1170/)

### Ethics & Bias
- *On the dangers of stochastic parrots: Can language models be too big?*, Bender et al. [[FAccT 2021]](https://dl.acm.org/doi/10.1145/3442188.3445922)
- *Data contamination: From memorization to exploitation*, Magar & Schwartz [[arXiv 2022]](https://arxiv.org/abs/2203.08242)
- *Evading data contamination detection: Exploring test-time preprocessing methods*, Dekoninck et al. [[arXiv 2024]](https://arxiv.org/abs/2405.20832)
- *Multimodal datasets: misogyny, pornography, and malignant stereotypes*, Birhane et al. [[NeurIPS 2021]](https://arxiv.org/abs/2110.01963)

### Theoretical Analysis & Understanding
- *Speak, memory: An archaeology of books known to chatgpt/gpt-4*, Chang et al. [[arXiv 2023]](https://arxiv.org/abs/2305.00118)
- *Faith and fate: Limits of transformers on compositionality*, Dziri et al. [[NeurIPS 2023]](https://arxiv.org/abs/2305.18654)
- *How much knowledge can you pack into the parameters of a language model?*, Roberts et al. [[EMNLP 2020]](https://aclanthology.org/2020.emnlp-main.437/)
- *Large language models struggle to learn long-tail knowledge*, Kandpal et al. [[arXiv 2023]](https://arxiv.org/abs/2211.08411)
- *A surprisingly robust trick for the winograd schema challenge*, Camburu et al. [[ACL 2019]](https://aclanthology.org/P19-1478/)
- *The curious case of neural text degeneration*, Holtzman et al. [[ICLR 2020]](https://arxiv.org/abs/1904.09751)
- *Neural text generation with unlikelihood training*, Welleck et al. [[ICLR 2020]](https://arxiv.org/abs/1908.04319)
- *Locally typical sampling*, Meister et al. [[TACL 2023]](https://aclanthology.org/2023.tacl-1.17/)
- *Information-theoretic probing for linguistic structure*, Pimentel et al. [[ACL 2020]](https://aclanthology.org/2020.acl-main.420/)
- *Predictability and surprise in large generative models*, Ganguli et al. [[arXiv 2022]](https://arxiv.org/abs/2202.07785)
- *A systematic evaluation of large language models of code*, Xu et al. [[arXiv 2022]](https://arxiv.org/abs/2202.13169)
- *Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning*, Liang et al. [[NeurIPS 2022]](https://arxiv.org/abs/2203.02053)
- *Well-tuned simple nets excel on tabular datasets*, Kadra et al. [[NeurIPS 2021]](https://arxiv.org/abs/2106.11189)

## Acknowledgments

We thank the open source community for providing access to the evaluated models via standardised interfaces. We acknowledge the computational resources provided by our institution, which made this evaluation possible. Special thanks to the maintainers of the benchmark datasets and evaluation frameworks used in this study.

## License

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.

---

*For more details, please refer to our [paper](https://arxiv.org/abs/2508.12461) and visit our [project website](https://github.com/ai-agent-lab/gpt-oss/).*

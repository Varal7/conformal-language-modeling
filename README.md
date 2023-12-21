# Conformal Language Modeling

Code for [Conformal Language Modeling](https://arxiv.org/abs/2306.10193) for details

## Abstract 

In this paper, we propose a novel approach to conformal prediction for language models (LMs) in which we produce prediction sets with performance guarantees. LM responses are typically sampled from a predicted distribution over the large, combinatorial output space of language. Translating this to conformal prediction, we calibrate a stopping rule for sampling LM outputs that get added to a growing set of candidates until we are confident that the set covers at least one acceptable response. Since some samples may be low-quality, we also simultaneously calibrate a rejection rule for removing candidates from the output set to reduce noise. Similar to conformal prediction, we can prove that the final output set obeys certain desirable distribution-free guarantees. Within these sets of candidate responses, we also show that we can also identify subsets of individual components---such as phrases or sentences---that are each independently correct (e.g., that are not "hallucinations"), again with guarantees. Our method can be applied to any LM API that supports sampling. Furthermore, we empirically demonstrate that we can achieve many desired coverage levels within a limited number of total samples when applying our method to multiple tasks in open-domain question answering, text summarization, and radiology report generation using different LM variants.

## Data

Also see our [auxiliary repo](https://github.com/Varal7/clm_aux) for data prepocessing.

## Citation

If you use this in your work please cite:

```
@misc{quach2023conformal,
      title={Conformal Language Modeling}, 
      author={Victor Quach and Adam Fisch and Tal Schuster and Adam Yala and Jae Ho Sohn and Tommi S. Jaakkola and Regina Barzilay},
      year={2023},
      eprint={2306.10193},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

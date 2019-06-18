# GSRNN

## Sample Dataset
flu data:
hhs.csv: size 650 by 10. flu count data of 650 weeks for 10 hhs regions (our nodes)
hhs_graph: the adjacency matrix of hhs regions

run Flu_Prediction.py to predict flu count.

This repo contains code for graph structured recurrent which can be used for general spatio-temporal data forecasting

If you find this work useful and use it on you own research, please cite our [paper](https://arxiv.org/abs/1811.10745)

```
@ARTICLE{Wang2018:RNN,
       author = {{B. Wang and X. Luo and F. Zhang and B. Yuan and A. L. Bertozzi and P. J. Brantingham},
        title = "{Graph-Based Deep Modeling and Real Time Forecasting of Sparse Spatio-Temporal Data}",
      journal = {arXiv e-prints},
         year = "2018",
        month = "April",
          eid = {arXiv:1804.00684},
        pages = {arXiv:1804.00684},
archivePrefix = {arXiv},
       eprint = {1804.00684},
 primaryClass = {stat.ML}
}
```

Another related article


```
@ARTICLE{GSRNN:Flu,
       author = {{Z. Li and X. Luo and B. Wang and A. L. Bertozzi and J. Xin},
        title = "{A Study on Graph-Structured Recurrent Neural Networks and Sparsification with Application to Epidemic Forecasting}",
      journal = {arXiv e-prints},
         year = "2019",
        month = "Feb",
          eid = {arXiv:1902.05113},
        pages = {arXiv:1902.05113},
archivePrefix = {arXiv},
       eprint = {1902.05113},
 primaryClass = {stat.ML}
}
```

## Dependence
Tensorflow

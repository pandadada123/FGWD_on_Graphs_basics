# FGW

Python3 implementation of the paper [Optimal Transport for structured data with application on graphs
](https://arxiv.org/abs/1805.09114) 

In the paper we used this implementation of Patchy-San convolutional network [PSCN](https://github.com/tvayer/PSCN)

### Prerequisites

* For graph tools [networkx](https://networkx.github.io/) Networkx (>=2.0)
* For the classification using Graph Kernels using [GraKel](https://ysig.github.io/GraKeL/dev/) GraKel (>=0.1a5) 
* For Optimal transport [Python Optimal Transport](https://pot.readthedocs.io/en/stable/) POT (>=0.5.1)
* Sklearn (>=0.20.0)
* Numpy (>=1.11)
* Scipy (>=1.0)
* Cython (>=0.23)
* Matplotlib (>=1.5)

### Data 

All the data used in the paper came from the Benchmark Data Sets for Graph Kernels [3]

### What is included ?

* FGW for euclidean measures 

![Alt text](coupling_on_1D.png)

* Comparing labeled graphs using FGW

<p align="center">
  <img src="https://github.com/tvayer/FGW/blob/master/coupling_on_graphs.png" width="340" >
</p>

* Methods for graphs barycenter using FGW 

![Alt text](barycircle.png)

* Nested cross validation used for benchmark data sets used in the paper :

```
python3 nested_cv_fgw.py -dn mutag -d ../data -r ../results -ni 10 -no 50  -fea hamming_dist -st shortest_path -cva True -wl 2 

```


### What will be added ?

* k-means of multiple graphs
* The Integration in the POT library [1] of FGW 

### Authors

* [Titouan Vayer](https://github.com/tvayer)
* [Laetitia Chapel](https://github.com/lchapel)
* [Romain Tavenard](https://github.com/rtavenar)
* [Nicolas Courty](https://github.com/ncourty)
* [Rémi Flamary](https://github.com/rflamary)


## References

[1] Flamary Rémi and Courty Nicolas [POT Python Optimal Transport library](https://github.com/rflamary/POT)

[2] Siglidis, Giannis and Nikolentzos, Giannis and Limnios, Stratis and Giatsidis, Christos and Skianis, Konstantinos and Vazirgiannis, Michali. [GraKeL: A Graph Kernel Library in Python](https://github.com/ysig/GraKeL)

[3] Kristian Kersting and Nils M. Kriege and Christopher Morris and Petra Mutzel and Marion Neumann [Benchmark Data Sets for Graph Kernels](http://graphkernels.cs.tu-dortmund.de)

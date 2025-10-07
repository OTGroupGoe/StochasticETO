## Overview
This repo provides code to generate data and to plot figures in [Entropic transfer operators for stochastic systems](https://www.arxiv.org/abs/2503.05308)

## Getting started
* [playground](./playground.ipynb) contains a simple example that shows the main spirit of the paper, also one can use it to generate plots in figure 3.
* [data_convection](./data_convection/) contains raw data from the lab, see section 3.5 in article for more details.
* get_results_convection.ipynb](./get_results_convection.ipynb) is used to simulate/process data that are required in the convection data plots, by default the results are saved to folder [results_convection](./results_convection/) (which is currently empty).

* [Given_results_torus](./Given_results_torus/) and [Given_results_convection](./Given_results_convection/) contain pre-generated data from us in case you do not have a cuda compatible gpu or to save your time.

* [torus.ipynb](./torus.ipynb) and [plot_figs_convection.ipynb](./plot_figs_convection.ipynb) are code for plotting the figures 4-8 and 9-13 respectively.

## License (MIT License)
Copyright (c) 2024, Hancheng Bi, Clément Sarrazin, Bernhard Schmitzer, Thilo Stier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

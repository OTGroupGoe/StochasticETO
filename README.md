## Overview
This repo provides code to generate data and to plot figures in "Entropic transfer operators for stochastic systems"

## Getting started
* [data_convection](./data_convection/) contains raw data from the lab, see section 3.3 in article for more details.
* [get_results_torus.ipynb](./get_results_torus.ipynb) and [get_results_convection.ipynb](./get_results_convection.ipynb) are used to simulate/process data that are required in the plots, by default the results are saved to folder [results_torus](./results_torus/) and [results_convection](./results_convection/) (which are currently empty) respectively. [get_results_convection.ipynb](./get_results_convection.ipynb) is suggested to run on machine with GPU.

* [Given_results_torus](./Given_results_torus/) and [Given_results_convection](./Given_results_convection/) contain pre-generated data from us to save your time.

* [plot_figs_torus.ipynb](./plot_figs_torus.ipynb) and [plot_figs_convection.ipynb](./plot_figs_convection.ipynb) are code for plotting the figures 2-5 and 6-10 respectively.

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

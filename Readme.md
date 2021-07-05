# Scientific Computing Seminar

Contains all the sources for the report, slides and the scripts.

As of now, the best way to use this repository is to `clone` it, navigate into the corresponding folder and

```Julia
using Pkg;
Pkg.activate()
Pkg.instantiate()
```

which should install all necessary packages into the corresponding environment. Afterwards, you can use the `ScientificComputing` package within the environment.

## Slides

To load the [Pluto Notebook](https://github.com/fonsp/Pluto.jl)

```Julia
using ScientificComputing
start_slides()
```

which starts the presentation in your browser.


## Scripts

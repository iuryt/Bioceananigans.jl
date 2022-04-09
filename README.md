
<!-- Title -->
<h1 align="center">
  Bioceananigans.jl
</h1>

<!-- description -->
<p align="center">
  <strong> Bringing life to physics on  <a href="https://github.com/CliMA/Oceananigans.jl">Oceananigans</a>.</strong>
</p>

There are many different ways to discretize relationships between different biogeochemical tracers in the ocean. One of those ways is the system of ordinary differential equations that is often called **N**utrient-**P**hytoplankton-**Z**ooplankton(-**D**etritus) set of models. Despite _simple_, these models could give insights about interelationships and biophysical interactions in ocean ecossystems across scales, from submesoscale to climate predictions.

Most of the applications of these set of models are either _offline_ to the physical part or intrinsic to the physical model infrastructure, usually written in FORTRAN language. This creates a problem because users many times doesn't understand well which equations are solved and cannot easily adapt to the set of equations they want to reproduce.

[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) is a very nice example on how numerical models could be **easy to use and adapt** without loosing the understanding behind the code. The usage of this physical ocean model is the closest to the human language that doesn't cross the black-box domain. Following the same principles of having something **as readable and configurable as possible**, this repository aims to basically create a set of modules and functions that are the essentials for building **NPZD models** and its derivatives on Oceananigans.

Most of the interelationships between biogeochemical tracers could be implemented using the **forcing functions** (see [Convecting Plankton](https://clima.github.io/OceananigansDocumentation/stable/generated/convecting_plankton/)  example from Oceananigans documentation), but commonly the light or light-limiting growth of the phytoplankton are **averaged over the mixed layer** in order to parameterized the effects of the mixing in the upper boundary layer. In other words, the mixing timescale is usually longer than photosynthesis, but shorter than cell-division time, which makes phytoplankton **see the average light or grow at the average light-limiting growth rate** while on the mixed layer. Other example of processes that are not easily implemented on Oceananigans include the **shading of phytoplankton** to deeper layers and the **sinking velocity**. The latter is currently being implemented on Oceananigans by this [Pull Request](https://github.com/CliMA/Oceananigans.jl/pull/2389) with a new forcing called `AdvectiveForcing`.

For now, this repository basically gives a set of modules that can be used to estimate the mixed-layer depth, estimate the phytoplankton shading and calculate the light-limiting growth, which is defined by the user as a function, and then the light (or the light-limiting growth) can be averaged over the mixed layer.

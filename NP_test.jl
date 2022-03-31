using Oceananigans
using Oceananigans.Units

# define the size and max depth of the simulation
const Ny = 100
const Nz = 48 # number of points in z
const H = 1000 # maximum depth

# create the grid of the model
grid = RectilinearGrid(GPU(),
    size=(Ny,Nz),
    y=(-(Ny/2)kilometers,(Ny/2)kilometers),
    z=(H * cos.(LinRange(π/2,0,Nz+1)) .- H)meters,
    topology=(Flat, Periodic, Bounded)
)

# define the turbulence closure of the model
horizontal_closure = ScalarDiffusivity(ν=1, κ=1)
vertical_closure = ScalarDiffusivity(ν=1e-5, κ=1e-5)

# constants for the NP model
const μ₀ = 1/day   # surface growth rate
const m = 0.015/day # mortality rate due to virus and zooplankton grazing

const Kw = 0.059 # meter^-1
const Kc = 0.041 # m^2 mg^-1
const kn = 0.75
const kr = 0.5
const α = 0.0538/day

# create the mld field that will be updated at every timestep
h = Field{Center, Center, Nothing}(grid) 
light = Field{Center, Center, Center}(grid)

const L0 = 100
# evolution of the available light at the surface
@inline L(z) = L0*exp.(z*Kw)
# light profile
@inline light_growth(z) = μ₀ * (L(z)*α)/sqrt(μ₀^2 + (L(z)*α)^2)

# nitrate and ammonium limiting
@inline N_lim(N,Nr) = (N/(N+kn)) * (kr/(Nr+kr))
@inline Nr_lim(Nr) =  (Nr/(Nr+kr))

# functions for the NP model
P_forcing(x, y, z, t, P, N, Nr)  =   light * (N_lim(N,Nr) + Nr_lim(Nr)) * P - m * P^2
N_forcing(x, y, z, t, P, N, Nr)  = - light * N_lim(N,Nr) * P
Nr_forcing(x, y, z, t, P, N, Nr) = - light * Nr_lim(Nr) * P + m * P^2

# using the functions to determine the forcing
P_dynamics = Forcing(P_forcing, field_dependencies = (:P,:N,:Nr))
N_dynamics = Forcing(N_forcing, field_dependencies = (:P,:N,:Nr))
Nr_dynamics = Forcing(Nr_forcing, field_dependencies = (:P,:N,:Nr))


coriolis = FPlane(latitude=60)

# create the model
model = NonhydrostaticModel(grid = grid,
                            coriolis = coriolis,
                            closure=(horizontal_closure,vertical_closure),
                            tracers = (:b, :P, :N, :Nr),
                            buoyancy = BuoyancyTracer(),
                            forcing = (P=P_dynamics,N=N_dynamics,Nr=Nr_dynamics))

# mld
const cz = -200
# gravity
const g = 9.82
# reference density
const ρₒ = 1026


# background density profile based on Argo data
@inline bg(z) = 0.25*tanh(0.0027*(-653.3-z))-6.8*z/1e5+1027.56

# decay function for fronts
@inline decay(z) = (tanh((z+500)/300)+1)/2

@inline front(x, y, z, cy) = tanh((y-cy)/12kilometers)
@inline D(x, y, z) = bg(z) + 0.8*decay(z)*front(x,y,z,0)/4
@inline B(x, y, z) = -(g/ρₒ)*D(x,y,z)

# initial phytoplankton profile
@inline P(x, y, z) = ifelse(z>cz,0.4,0)

# setting the initial conditions
set!(model;b=B,P=P,N=13,Nr=0)


simulation = Simulation(model, Δt = 1minutes, stop_time = 2minutes)


# # create a simulation
# simulation = Simulation(model, Δt = 1minutes, stop_time = 20days)

# wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=6minutes)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

include("src/compute_mixed_layer_depth.jl")
const Δb=(g/ρₒ) * 0.03
compute_mixed_layer_depth!(simulation) = compute_mixed_layer_depth!(h, simulation.model.tracers.b, Δb)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)

include("src/compute_light.jl")
compute_light!(simulation) = compute_light!(light, h, simulation.model.tracers.P, light_growth)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_light] = Callback(compute_light!)

# merge light and h to the outputs
outputs = merge(model.velocities, model.tracers, (; light, h)) # make a NamedTuple with all outputs
# writing the output
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, outputs, filepath = "data/NP_output.nc",
                     schedule=TimeInterval(1hour))

using Printf

function print_progress(simulation)
    b, P, N, Nr = simulation.model.tracers

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, P_max = %.1e, N_min = %.1e, Nr_max = %.1e, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(P), minimum(N), maximum(Nr),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

# simulation.callbacks[:progress] = Callback(print_progress, TimeInterval(1hour))
simulation.callbacks[:progress] = Callback(print_progress, IterationInterval(1))

# run the simulation
run!(simulation)
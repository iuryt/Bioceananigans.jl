using Oceananigans
using Oceananigans.Units

# define the size and max depth of the simulation
const Nx = 2
const Ny = 2
const Nz = 48 # number of points in z
const H = 1000 # maximum depth

# create the grid of the model
grid = RectilinearGrid(GPU(),
    size=(Nx,Ny,Nz),
    x=(-Nx/2,Nx/2),
    y=(-Ny/2,Ny/2),
    z=(H * cos.(LinRange(π/2,0,Nz+1)) .- H)meters,
    topology=(Periodic, Periodic, Bounded)
)

# define the turbulence closure of the model
closure = ScalarDiffusivity(ν=1e-5, κ=1e-5)

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

#@inline light_growth(light) = (light*α)/sqrt(μ₀^2 + (light*α)^2)

const L0 = 100
# evolution of the available light at the surface
@inline L(z) = L0*exp.(z*Kw)
# light profile
@inline light_growth(z,t) = μ₀ * (L(z)*α)/sqrt(μ₀^2 + (L(z)*α)^2)

# nitrate and ammonium limiting
@inline N_lim(N,Nr) = (N/(N+kn)) * (kr/(Nr+kr))
@inline Nr_lim(Nr) =  (Nr/(Nr+kr))

# functions for the NP model
P_forcing(x, y, z, t, P, N, Nr)  =   light_growth(z,t) * (N_lim(N,Nr) + Nr_lim(Nr)) * P - m * P^2
N_forcing(x, y, z, t, P, N, Nr)  = - light_growth(z,t) * N_lim(N,Nr) * P
Nr_forcing(x, y, z, t, P, N, Nr) = - light_growth(z,t) * Nr_lim(Nr) * P + m * P^2

# using the functions to determine the forcing
P_dynamics = Forcing(P_forcing, field_dependencies = (:P,:N,:Nr))
N_dynamics = Forcing(N_forcing, field_dependencies = (:P,:N,:Nr))
Nr_dynamics = Forcing(Nr_forcing, field_dependencies = (:P,:N,:Nr))

# create the model
model = NonhydrostaticModel(grid = grid,
                            closure=closure,
                            tracers = (:b, :P, :N, :Nr),
                            buoyancy = BuoyancyTracer(),
                            forcing = (P=P_dynamics,N=N_dynamics,Nr=Nr_dynamics))

# mld
const cz = -200
# gravity
const g = 9.82
# reference density
const ρₒ = 1026

# initial buoyancy profile based on Argo data
@inline B(x, y, z) = -(g/ρₒ)*0.25*tanh(0.0027*(-653.3-z))-6.8*z/1e5+1027.56
# initial phytoplankton profile
@inline P(x, y, z) = ifelse(z>cz,0.4,0)

# setting the initial conditions
set!(model;b=B,P=P,N=13,Nr=0)

# create a simulation
simulation = Simulation(model, Δt = 1minutes, stop_time = 20days)

using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!

@kernel function _compute_mixed_layer_depth(h, grid, b, Δb)
    i, j = @index(Global, NTuple)

    # Use this to set mld to bottom (note for irregular domains, this would have to change)
    z_bottom = znode(Center(), Center(), Face(), i, j, 1, grid)
    b_surface = @inbounds b[i, j, grid.Nz] # buoyancy at surface

    @unroll for k in grid.Nz - 1 : -1 : 1 # scroll to point just above bottom
        b_ijk = @inbounds b[i, j, k]
        below_mixed_layer = @inbounds b_surface - b_ijk > Δb

        # height of the cell interface _above_ the current depth
        # TODO: use linear interpolation to obtain a smooth measure of mixed layer depth instead?
        z_face_above = znode(Center(), Center(), Face(), i, j, k+1, grid)
        z_below_mixed_layer_ij = ifelse(below_mixed_layer, z_face_above, z_bottom)
    end

    # Note "-" since `h` is supposed to be "depth" rather than "height"
    @inbounds h[i, j, 1] = - z_below_mixed_layer_ij
end

function compute_mixed_layer_depth!(h, b, Δb)
    grid = h.grid
    arch = architecture(grid)

    event = launch!(arch, grid, :xy,
                    _compute_mixed_layer_depth!, h, grid, b, Δb,
                    dependencies = device_event(arch))

    wait(device_event(arch), event)

    return nothing
end

# add the function to the callbacks of the simulation
simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)

# function to update the light profile (not working)
function compute_light_profile!(sim)
    x,y,z = nodes((Center,Center,Center),sim.model.grid)
    
    dz = z[2:end]-z[1:end-1]
    dz = [dz[1];dz]
    
    for i=1:length(x)
        for j=1:length(y)
            Li = L0(time(sim))*exp.(z*Kw)
            
            inmld = z.>h[i,j]
            Li[inmld] .= sum(Li[inmld].*dz[inmld])/abs(h[i,j])
            
            for k=1:length(z)
                light[i,j,k] = Li[k]
            end
        end
    end
    return nothing
end
# simulation.callbacks[:compute_light] = Callback(compute_light_profile!)


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

simulation.callbacks[:progress] = Callback(print_progress, TimeInterval(1hour))

# run the simulation
run!(simulation)

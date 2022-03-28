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

#light = Field{Center, Center, Center}(grid)
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

# function to compute the mld
function compute_mixed_layer_depth!(sim)
    # get the buoyancy from the simulation
    b = sim.model.tracers.b
    # extract the model nodes
    x,y,z = nodes((Center,Center,Center),sim.model.grid)
    
    # loop by x-axis
    for i=1:length(x)
        # loop by y axis
        for j=1:length(y)
            # get the density profile at (i,j)
            ρⁿ = -(ρₒ/g)*Array(interior(b, i, j, :))
            # surface density
            ρs = ρⁿ[end]
            # criterion for density increment
            dρ = 0.01
            # loop backwards (surface->bottom)
            for k=0:length(z)-1
                # if the density satisfies the criterium
                if ρⁿ[end-k]>=ρs+dρ
                    # update mld values
                    h[i,j]=z[end-k]
                    # break the loop
                    break
                end
            end
        end
    end
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

# writing the output
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, merge(model.velocities, model.tracers), filepath = "data/NP_output.nc",
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

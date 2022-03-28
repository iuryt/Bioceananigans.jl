using Oceananigans
using Oceananigans.Units

const Nx = 5
const Ny = 5
const Nz = 48 # number of points in z
const H = 1000 # maximum depth


grid = RectilinearGrid(CPU(),
    size=(Nx,Ny,Nz),
    x=(-Nx/2,Nx/2),
    y=(-Ny/2,Ny/2),
    z=(H * cos.(LinRange(π/2,0,Nz+1)) .- H)meters,
    topology=(Periodic, Periodic, Bounded)
)



closure = ScalarDiffusivity(ν=1e-5, κ=1e-5)


const μ₀ = 1/day   # surface growth rate
const m = 0.015/day # mortality rate due to virus and zooplankton grazing

const Kw = 0.059 # meter^-1
const Kc = 0.041 # m^2 mg^-1
const kn = 0.75
const kr = 0.5
const α = 0.0538/day

h = Field{Center, Center, Nothing}(grid) # defines a field that's reduced in the vertical direction

light = Field{Center, Center, Center}(grid)
@inline L0(t) = t/10 + 14

@inline light_growth(light) = (light*α)/sqrt(μ₀^2 + (light*α)^2)
@inline N_lim(N,Nr) = N/(N+kn) * kr/(Nr+kr)
@inline Nr_lim(Nr) = Nr/(Nr+kr)

P_forcing(x, y, z, t, P, N, Nr)  =   μ₀ * light_growth(light) * (N_lim(N,Nr) + Nr_lim(Nr)) * P - m * P^2
N_forcing(x, y, z, t, P, N, Nr)  = - μ₀ * light_growth(light) * N_lim(N,Nr) * P
Nr_forcing(x, y, z, t, P, N, Nr) = - μ₀ * light_growth(light) * N_lim(N,Nr) * P + m * P^2


P_dynamics = Forcing(P_forcing, field_dependencies = (:P,:N,:Nr))
N_dynamics = Forcing(N_forcing, field_dependencies = (:P,:N,:Nr))
Nr_dynamics = Forcing(Nr_forcing, field_dependencies = (:P,:N,:Nr))

model = NonhydrostaticModel(grid = grid,
                            closure=closure,
                            tracers = (:b, :P, :N, :Nr),
                            buoyancy = BuoyancyTracer(),
                            forcing = (P=P_dynamics,N=N_dynamics,Nr=Nr_dynamics))


const cz = -250
const g = 9.82
const ρₒ = 1026

# background density profile based on Argo data
@inline bg(z) = 0.25*tanh(0.0027*(-653.3-z))-6.8*z/1e5+1027.56
@inline B(x, y, z) = -(g/ρₒ)*bg(z)

@inline P(x, y, z) = ifelse(z>cz,0.4,0)

set!(model;b=B,P=P,N=13,Nr=0)


simulation = Simulation(model, Δt = 1minutes, stop_time = 10minutes)


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


function compute_mixed_layer_depth!(sim)
    b = sim.model.tracers.b
    x,y,z = nodes((Center,Center,Center),sim.model.grid)
    for i=1:length(x)
        for j=1:length(y)
            ρⁿ = -(ρₒ/g)*Array(interior(b, i, j, :))
            ρs = ρⁿ[end]
            dρ = 0.01
            for k=0:length(z)-1
                if ρⁿ[end-k]>=ρs+dρ
                    h[i,j]=z[end-k]
                    break
                end
            end
        end
    end
    return nothing
end

simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)
simulation.callbacks[:compute_light] = Callback(compute_light_profile!)

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, merge(model.velocities, model.tracers), filepath = "data/NP_output.nc",
                     schedule=TimeInterval(1minute))


run!(simulation)

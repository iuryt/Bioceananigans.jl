using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids
using Oceananigans.Operators: Δzᶜᶜᶜ

@kernel function _compute_light_growth!(light, grid, h, P, light_function, light_growth_function, simulation_time, 
                                         average_code, shading, chl2c, Kc)
    i, j = @index(Global, NTuple)

    h_ij = @inbounds h[i, j]
    
    chlinteg = 0
    
    @unroll for k in grid.Nz : -1 : 1
        if shading
            chlinteg = chlinteg + @inbounds P[i, j, k] * chl2c * Δzᶜᶜᶜ(i, j, k, grid)
        end
        z_center = znode(Center(), Center(), Center(), i, j, k, grid)
        
        if average_code==1
            @inbounds light[i, j, k] = light_growth_function(light_function(simulation_time, z_center)/exp(chlinteg*Kc))
        elseif (average_code==0)|(average_code==2)
            @inbounds light[i, j, k] = light_function(simulation_time, z_center)/exp(chlinteg*Kc)
        end
        
    end

    if ~(average_code==0)
        light_sum = 0
        dz_sum = 0

        @unroll for k in grid.Nz : -1 : 1 # scroll from surface to bottom       
            z_center = znode(Center(), Center(), Center(), i, j, k, grid)

            if z_center > - h_ij

                Δz_ijk = Δzᶜᶜᶜ(i, j, k, grid)

                light_sum = light_sum + @inbounds light[i, j, k] * Δz_ijk 
                dz_sum = dz_sum + Δz_ijk

            end
        end

        @unroll for k in grid.Nz : -1 : 1 # scroll to point just above the bottom       
            z_center = znode(Center(), Center(), Center(), i, j, k, grid)

            if z_center > - h_ij
                @inbounds light[i, j, k] = light_sum/dz_sum     
            end
        end
    end
    
    if average_code!=:1
        @unroll for k in grid.Nz : -1 : 1
            @inbounds light[i, j, k] = light_growth_function(light[i, j, k])
        end
    end
    
end

"""
    LightGrowth!(light, h, P, light_function, light_growth_function, simulation_time, average=nothing, shading=false, chl2c=0.02, Kc = 0.041)

Compute and update the light growth for Phytoplankton.
"""
function LightGrowth!(light, h, P, light_function, light_growth_function, simulation_time, 
                      average=nothing, shading=false, chl2c=0.02, Kc = 0.041)
    grid = h.grid
    arch = architecture(grid)

    
    if average==nothing
        average_code = 0
    elseif average==:growth
        average_code = 1
    elseif average==:light
        average_code = 2
    else
        error("Unsupported average ($average).")
    end
    
    event = launch!(arch, grid, :xy,
                    _compute_light_growth!, light, grid, h, P, light_function, light_growth_function, 
                    simulation_time, average_code, shading, chl2c, Kc,
                    dependencies = device_event(arch))

    wait(device_event(arch), event)

    return nothing
end
using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids

@kernel function _compute_light!(light, grid, h, P, light_function)
    i, j = @index(Global, NTuple)

    @unroll for k in 1 : grid.Nz
        z_center = znode(Center(), Center(), Center(), i, j, k, grid)
        @inbounds light[i, j, k] = light_function(z_center)
    end

    light_sum = 0
    dz_sum = 0
    
    @unroll for k in grid.Nz-1 : -1 : 1 # scroll to point just above the bottom       
        z_center = znode(Center(), Center(), Center(), i, j, k, grid)
        z_center_above = znode(Center(), Center(), Center(), i, j, k-1, grid) 
        
        h_ijk = @inbounds -h[i, j, k]
        
        if z_center>h_ijk
            
            light_ijk = @inbounds light[i, j, k]
            Δz_ijk = z_center_above-z_center
            
            light_sum += light_ijk*Δz_ijk
            dz_sum += Δz_ijk
            
        end
    end
    
    @unroll for k in grid.Nz : -1 : 1 # scroll to point just above the bottom       
        z_center = znode(Center(), Center(), Center(), i, j, k, grid)
        
        h_ijk = @inbounds h[i, j, k]
        
        if z_center > -h_ijk
            @inbounds light[i, j, k] = light_sum/dz_sum     
        end
    end
end

function compute_light!(light, h, P, light_function)
    grid = h.grid
    arch = architecture(grid)

    event = launch!(arch, grid, :xy,
                    _compute_light!, light, grid, h, P, light_function,
                    dependencies = device_event(arch))

    wait(device_event(arch), event)

    return nothing
end

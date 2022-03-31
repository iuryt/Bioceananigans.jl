using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids

@kernel function _compute_mixed_layer_depth!(h, grid, b, Δb)
    i, j = @index(Global, NTuple)

    # Use this to set mld to surface
    z_surface = znode(Center(), Center(), Face(), i, j, grid.Nz, grid)
    b_surface = @inbounds b[i, j, grid.Nz] # buoyancy at surface
    
    # at the first iteration, mld is at the surface
    mld_ij = z_surface
    searching_mld = true
    
    @unroll for k in grid.Nz -1 : -1 : 1 # scroll from point just below surface
        # buoyancy and buoyancy decrease
        b_ijk = @inbounds b[i, j, k]
        Δb_ijk = b_surface-b_ijk
        
        # if below mixed layer and we havent yet found it
        if (Δb_ijk>Δb)&(searching_mld)
            # buoyancy and buoyancy decrease for the cell above
            b_ijk_above = @inbounds b[i, j, k-1]
            Δb_ijk_above = b_surface-b_ijk_above
            
            # height for both cells
            z_face = znode(Center(), Center(), Face(), i, j, k, grid)
            z_face_above = znode(Center(), Center(), Face(), i, j, k-1, grid)
            
            # get Δz and Δb
            Δz_ijk = z_face_above-z_face
            
            # interpolate linearly to obtain the mld
            mld_ij = -((Δz_ijk/(Δb_ijk_above-Δb_ijk))*(Δb-Δb_ijk)+z_face)
            
            # stop searching (skip this calculation for other levels)
            searching_mld=false
        end
    end

    # Note "-" since `h` is supposed to be "depth" rather than "height"
    @inbounds h[i, j, 1] = mld_ij
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

#include "common.h"
#include <mpi.h>
#include <math.h>
#include <algorithm> 
#include <cstring>
#include <vector> 

// Put any static global variables here that you will use throughout the simulation.
#define STOPTAG 200000

typedef struct {            // structure to represent a bin
    int neighbor_id[8];     // index ids of neighboring bins
    int particle_id[8];     // index ids of particles in this bin
    int num_neighbors;      // actual number of neighbors
    int num_particles;      // actual number of particles
} bin_t;

int bins_per_side;          // number of bins per side
double bin_size;            // size of bin
int num_bins;               // size of bin
bin_t *bins;
int *bin_offsets;
int *bin_sizes;
double bin_size_reciprocal;

// From provided HW2-1 serial code:
// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Inverse sqrt
    double r_inv = 1.0 / sqrt(r2);      // single sqrt
    double r2_inv = r_inv * r_inv;      // no sqrt for r2
    double cutoff_r = cutoff * r_inv;   // cutoff / r
    double coef = (1.0 - cutoff_r) * r2_inv / mass;

    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

/*
Sets up a grid of bins for particle simulation, 
assigning neighbors to each bin while excluding the bin itself. 
Also initialize particle counts within each bin, preparing for simulation calculations.
*/
void init_bins(bin_t *bins) {
    for (int i = 0; i < bins_per_side; ++i) {
        for (int j = 0; j < bins_per_side; ++j) {
            int bin_index = i * bins_per_side + j;
            bin_t &bin = bins[bin_index];
            int num_neighbors = 0;
            bin.num_particles = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int neighbor_i = i + di;
                    int neighbor_j = j + dj;
                    if (neighbor_i >= 0 && neighbor_i < bins_per_side &&
                        neighbor_j >= 0 && neighbor_j < bins_per_side &&
                        (di != 0 || dj != 0)) 
                    {
                        int neighbor_index = neighbor_i * bins_per_side + neighbor_j;
                        bin.neighbor_id[num_neighbors] = neighbor_index;
                        num_neighbors++;
                    }
                }
            }
            bin.num_neighbors = num_neighbors;
        }
    }
}

/*
Assign particles to bins based on their positions, 
using a reciprocal multiplication for efficiency.
Update the count of particles in each bin accordingly.
*/
void assign_to_bins(bin_t *bins, particle_t *particles, int n) {
    bin_size_reciprocal = 1.0 / bin_size;
    // assign each particle to the appropriate bin
    for(int i = 0; i < n; i++) {
        int x = int(particles[i].x * bin_size_reciprocal);
        int y = int(particles[i].y * bin_size_reciprocal);
        int bin_index = y * bins_per_side + x;
        bins[bin_index].particle_id[bins[bin_index].num_particles++] = i;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Initialize bins
    bins_per_side = ceil(size / (cutoff * 1.8));    // TODO: tune cut-off
    bin_size = size / bins_per_side;
    num_bins = bins_per_side * bins_per_side;
    bins = (bin_t*) malloc(num_bins * sizeof(bin_t));
    init_bins(bins);
    
    // Set up bin partitioning across processors
    int columns_per_proc = bins_per_side / num_procs;
    int remainder_columns = bins_per_side % num_procs;

    bin_offsets = (int*) malloc((num_procs + 1) * sizeof(int));
    bin_sizes = (int*) malloc(num_procs * sizeof(int));

    for (int i = 0; i < num_procs; i++) {
        bin_offsets[i] = i * columns_per_proc * bins_per_side;
        // if the last process, add the remainder columns
        if (i == num_procs - 1) {
            // Last process takes its share plus any remaining columns
            bin_sizes[i] = (columns_per_proc + remainder_columns) * bins_per_side;
        } else {
            // Other processes just take their share of columns
            bin_sizes[i] = columns_per_proc * bins_per_side;
        }
    }
    // last offset
    bin_offsets[num_procs] = num_bins;

    // column start and end for current process
    double col_start = columns_per_proc * rank * bin_size;
    double col_end = columns_per_proc * (rank+1) * bin_size;
    if (rank == num_procs - 1) col_end = size;
    assign_to_bins(bins, parts, num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
    // set up the bin partitioning across processors
    int columns_per_proc = bins_per_side / num_procs;

    // column start and end for current process
    double col_start = columns_per_proc * rank * bin_size;
    double col_end = columns_per_proc * (rank+1) * bin_size;

    // record local particles
    int local_particles[4*bin_sizes[rank]];
    int local_particle_size = 0;

    // Compute forces
    for (int i = bin_offsets[rank]; i < bin_offsets[rank+1]; ++i) {
        bin_t &bin = bins[i];
        // Iterate over particles in current bin
        for (int j = 0; j < bin.num_particles; ++j) {
            local_particles[local_particle_size++] = bin.particle_id[j];
            particle_t &p = parts[bin.particle_id[j]];
            p.ax = p.ay = 0;
            // Apply forces from particles within the same bin, excluding self
            for (int q = 0; q < bin.num_particles; ++q) {
                if (q != j) {
                    apply_force(p, parts[bin.particle_id[q]]);     
                }
            }
            // Apply forces from particles in neighboring bins
            for (int k = 0; k < bin.num_neighbors; ++k) {
                bin_t &neighbor_bin = bins[bin.neighbor_id[k]];
                for (int m = 0; m < neighbor_bin.num_particles; ++m) {
                    apply_force(p, parts[neighbor_bin.particle_id[m]]);     
                }
            }
        }
    }

    // init allocated bins + ghost zone
    int init_start = (rank == 0) ? bin_offsets[rank] : bin_offsets[rank] - bins_per_side;
    int init_end = (rank == num_procs-1) ? bin_offsets[rank+1] : bin_offsets[rank+1] + bins_per_side;
    for (int i = init_start; i < init_end; ++i) {
        bins[i].num_particles = 0;
    }

    // move particles
    for( int i = 0; i < local_particle_size; i++ ) {
        int p = local_particles[i];
        move(parts[p], size);
        // after moving, reassign each particle to the appropriate bin
        double col = parts[p].y;
        if (col < col_start || col >= col_end) {
            int target = col < col_start ? rank - 1:rank + 1;
            MPI_Request request;
            if (target >= 0 && target < num_procs) {
                MPI_Isend(&parts[p], 1, PARTICLE, target, local_particles[i], MPI_COMM_WORLD, &request);
            }
        } else {
            int x = int(parts[p].x * bin_size_reciprocal);
            int y = int(parts[p].y * bin_size_reciprocal);
            int bin_index = y * bins_per_side + x;
            bin_t &bin = bins[bin_index];
            bin.particle_id[bin.num_particles++] = p;
        }
    }
    // send ghost zones to neighbors
    if (num_procs > 1) {
        MPI_Request ghost_request;
        if(rank != 0) {
            for (int i = bin_offsets[rank]; i < bin_offsets[rank] + bins_per_side; ++i) {
                bin_t &bin = bins[i];
                for (int j = 0; j < bin.num_particles; ++j) {
                    MPI_Isend(&parts[bin.particle_id[j]], 1, PARTICLE, rank - 1, bin.particle_id[j], MPI_COMM_WORLD, &ghost_request);
                }
            }
        }
        if(rank != num_procs-1) {
            for (int i = bin_offsets[rank+1]-bins_per_side; i < bin_offsets[rank+1]; ++i) {
                bin_t &bin = bins[i];
                for (int j = 0; j < bin.num_particles; ++j) {
                    MPI_Isend(&parts[bin.particle_id[j]], 1, PARTICLE, rank + 1, bin.particle_id[j], MPI_COMM_WORLD, &ghost_request);
                }
            }
        }
    }

    // send terminating message to neighbors
    if (num_procs > 1) {
        MPI_Request stop_request;
        if (rank != 0) {
            MPI_Isend(0, 0, MPI_INT, rank - 1, STOPTAG, MPI_COMM_WORLD, &stop_request);
        }
        if (rank != num_procs - 1) {
            MPI_Isend(0, 0, MPI_INT, rank + 1, STOPTAG, MPI_COMM_WORLD, &stop_request);
        }
    }
    int recv_count = 0;
    int recv_max = (rank == 0 || rank == num_procs - 1) ? 1 : 2;

    MPI_Status status;
    while(recv_count < recv_max && num_procs > 1) {
        particle_t tmp;
        MPI_Recv(&tmp, 1, PARTICLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int i = status.MPI_TAG;
        if (i == STOPTAG ) {
            recv_count ++;
            continue;
        }
        parts[i] = tmp;
        
        int x = int(tmp.x * bin_size_reciprocal);
        int y = int(tmp.y * bin_size_reciprocal);
        int bin_index = y * bins_per_side + x;
        bin_t &bin = bins[bin_index];
        bin.particle_id[bin.num_particles++] = i;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    if (rank == 0) {
        // TODO: 
    }
}
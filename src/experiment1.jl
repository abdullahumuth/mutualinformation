include("./main.jl")

name = "ER_improved_uniform_sample_convergence_l20_batch128"
version = 1

experiment1 = experiment(L, J, g, 0.5:0.5:1.0, (2^x for x=9:16))
experiment1(name, version; )





# test = experiment(L, J, g, 0.1, 100)
# test("test", 1, new = true)


# sample_experiment = experiment(L, J, g, 0.5:0.5:1.0, (2^x for x=9:16))
# sample_experiment("unique_uniform_sample_convergence_l20_batch128", 1; )
# sample_experiment("transfer_sample_convergence_large_batch128", 1; new = true)
# sample_experiment("sample_convergence_large_batch256", 1; batch_size = 256)
# sample_experiment("sample_convergence_large_batch512", 1; batch_size = 512)
# sample_experiment("sample_convergence_large_batch1024", 1; batch_size = 1024)
# sample_experiment("transfer_sample_convergence_large_batch1024", 1; batch_size = 1024, new = true)

# time_evolve_experiment = experiment(20, J, g, 0.0:0.05:1.0, 10000)
# 
# time_evolve_experiment("unique_uniform_time_evolve_l20_batch128", 1, batch_size = 128)
# time_evolve_experiment("transfer_time_evolve_convergence_large_batch256", 1; new = true, batch_size = 256)



# moment_of_truth = experiment(L, J, g, 0.0, (2^x for x=9:16))
# moment_of_truth("moment_of_truth_batch128_1partition", 1; batch_size = 128)
import subprocess

# num_epochs_values = [
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0002.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0003.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0004.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0005.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0006.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0007.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0008.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0009.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0010.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0011.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0012.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0013.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0014.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0015.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0016.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0017.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0018.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0019.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0020.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0021.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0022.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0023.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0024.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0025.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0026.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0027.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0028.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0029.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0030.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0031.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0032.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0033.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0034.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0035.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0036.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0037.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0038.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0039.h5',
# '/home/njucm/下载/H5-Data/CPDB_multiomics_samsplit_0040.h5',
# ]

num_epochs_values = ['/home/njucm/下载/CPDB_multiomics_samsplit_luad_lusc.h5']

for num_epochs_value in num_epochs_values:
    print(f"当前程序运行到{num_epochs_value}\n")
    arguments = [
        "python",
        "/home/njucm/emogi-reusability-main/HERBGAT.py",
        "--num_epochs=1500",
        "--hidden_dims=64",
        "--heads=4",
        "--dropout=0.2",
        "--loss_mul=1",
        "--sample_filename="+num_epochs_value,
        "--lr=0.001",
        "--seed=1",
        "--cuda"
    ]

    subprocess.run(arguments)

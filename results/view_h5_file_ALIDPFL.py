import h5py
import matplotlib.pyplot as plt


file_path = 'mnist_ALIDPFL_train_Rs_536.h5'

with h5py.File(file_path, 'r') as hf:

    rs_test_acc_data = hf['rs_test_acc'][:]
    rs_train_loss_data = hf['rs_train_loss'][:]
    rs_server_acc_data = hf['rs_server_acc'][:]
    rs_server_loss_data = hf['rs_server_loss'][:]
    rs_tau_list = hf['rs_tau_list'][:]
    extra_msg = hf['extra_msg'][()]

    extra_msg = extra_msg.decode('utf-8')
    print("----------------------------------------------")
    print(extra_msg)
    print("rs_server_acc:", rs_server_acc_data)
    print("best_server_acc", max(rs_server_acc_data))
    print("rs_server_loss:", rs_server_loss_data)
    print("rs_tau_list:", rs_tau_list)
    print("----------------------------------------------")


    print("================================================================")

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # 1：rs_test_acc_data和rs_server_acc_data
    axs[0].plot(rs_test_acc_data, label='rs_test_acc')
    axs[0].plot(rs_server_acc_data, label='rs_server_acc')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Value')
    axs[0].set_title(file_path)
    axs[0].legend()

    # 2：rs_train_loss_data和rs_server_loss_data
    axs[1].plot(rs_train_loss_data, label='rs_train_loss')
    axs[1].plot(rs_server_loss_data, label='rs_server_loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Value')
    axs[1].set_title(file_path)
    axs[1].legend()

    # 3：rs_tau_list
    axs[2].plot(rs_tau_list, label='rs_tau_list')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Value')
    axs[2].set_title(file_path)
    axs[2].legend()

    plt.tight_layout()

    plt.show()

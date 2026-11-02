import qcd_ml
import numpy as np
import torch
import matplotlib.pylab as plt
from tqdm import tqdm
import argparse
import models
from configs import Config
from aquarel import load_theme
theme = load_theme("umbra_dark")
theme.apply()


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

config = Config()

def load_data():

    U_1200 = torch.tensor(np.load("./qcd_ml/test/assets/1200.config.npy")).to(device)
    U_1500 = torch.tensor(np.load("./qcd_ml/test/assets/1500.config.npy")).to(device)
    U_2000 = torch.tensor(np.load("./qcd_ml/test/assets/2000.config.npy")).to(device)
    U_2200 = torch.tensor(np.load("./qcd_ml/test/assets/2200.config.npy")).to(device)

    U_510 = torch.load("./gauge fields/ens_4c8_beta6.0_001/510.pt").to(device)
    U_520 = torch.load("./gauge fields/ens_4c8_beta6.0_001/520.pt").to(device)
    U_530 = torch.load("./gauge fields/ens_4c8_beta6.0_001/530.pt").to(device)
    U_540 = torch.load("./gauge fields/ens_4c8_beta6.0_001/540.pt").to(device)


    # specify U's for testing and training and generating used dirac operators

    global U_training_set
    global U_testing_set

    #U_training_set = (U_1200, U_1500, U_2000)
    #U_testing_set = (U_2200,)

    U_training_set = (U_510, U_520, U_530)
    U_testing_set = (U_540,)   


def create_dirac_operator(type: str, m, csw = 1):

    global dirac_testing_set
    global dirac_training_set

    if type == "DWC":

        dirac_training_set = tuple(qcd_ml.qcd.dirac.dirac_wilson_clover(U, m, csw)
            for U in U_training_set
        )

        dirac_testing_set = tuple(qcd_ml.qcd.dirac.dirac_wilson_clover(U, m, csw)
            for U in U_testing_set
        )

    if type == "DW":
                
        dirac_training_set = tuple(qcd_ml.qcd.dirac.dirac_wilson(U, m)
            for U in U_training_set
        )

        dirac_testing_set = tuple(qcd_ml.qcd.dirac.dirac_wilson(U, m)
            for U in U_testing_set
        )



def path_generator(type: int):

    zero_hop_paths = [[]]
    one_hop_paths = [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]

    circular_paths = []

    for mu in range(4):
        for nu in range(4):
            if nu == mu:
                continue
            circular_paths.append([(mu, 1), (nu, 1), (mu, -1), (nu, -1)])
            circular_paths.append([(mu, 1), (nu, -1), (mu, -1), (nu, 1)])
            circular_paths.append([(mu, -1), (nu, 1), (mu, 1), (nu, -1)])
            circular_paths.append([(mu, -1), (nu, -1), (mu, 1), (nu, 1)])

    match type:
        case 1:
            return zero_hop_paths
        case 2:
            return zero_hop_paths + one_hop_paths
        case 3:
            return zero_hop_paths + one_hop_paths + circular_paths



def random_norm_quark_field(config):

    v = torch.randn(config.Lx, config.Ly, config.Lz, config.Lt, 4, 3, device=device, dtype=torch.cdouble)
    v_normalized = v / qcd_ml.util.linear_algebra.norm(v)

    return v_normalized



# training by calculating cost over set of U for set training steps

def training_loop(config, model, optimizer, training_steps):
    global loss_history 
    loss_history = []

    for i in tqdm(range(training_steps)):

        optimizer.zero_grad()
        costs = []

        for U, dirac in zip(U_training_set, dirac_training_set):

            v = random_norm_quark_field(config)
            w = dirac(v)
            w = model(w, U)

            cost_per_step = qcd_ml.util.linear_algebra.innerproduct(w-v, w-v).real
            costs.append(cost_per_step)

        cost = torch.stack(costs).mean()

        cost.backward()
        optimizer.step()
      
        loss_history.append(cost.item())

    print("Final training cost:", cost)

def training_loop_lowfeedback(config, model, optimizer, training_steps):

    for i in range(training_steps):

        optimizer.zero_grad()
        costs = []

        for U, dirac in zip(U_training_set, dirac_training_set):

            v = random_norm_quark_field(config)
            w = dirac(v)
            w = model(w, U)

            cost_per_step = qcd_ml.util.linear_algebra.innerproduct(w-v, w-v).real
            costs.append(cost_per_step)

        cost = torch.stack(costs).mean()

        cost.backward()
        optimizer.step()


def training_loop_auto(config, model, optimizer, threshold, max_training_steps):

    old_cost = None

    for training_step in range(1, max_training_steps + 1):

        optimizer.zero_grad()
        costs = []

        for U, dirac in zip(U_training_set, dirac_training_set):

            v = random_norm_quark_field(config)
            w = dirac(v)
            w = model(w, U)

            cost_per_step = qcd_ml.util.linear_algebra.innerproduct(w-v, w-v).real
            costs.append(cost_per_step)

        cost = torch.stack(costs).mean()

        cost.backward()
        optimizer.step()

        current_cost = cost.item()

        if old_cost is not None and old_cost > 0:
            improvement = (old_cost - current_cost) / old_cost

            if 0 <= improvement < threshold:

                print(f"Converged after {training_step} steps.")
                break

        old_cost = current_cost

    print("Final training cost:", current_cost)



# testing using four gauge fields using 100 random normalized quark fields for each and mean, std across all msrments

def testing_loop(config, model, sample_size):

    approx_history = []

    with torch.no_grad():

        for U, dirac in zip(U_testing_set, dirac_testing_set):

            for i in range(sample_size):
                
                v = random_norm_quark_field(config)
                w = dirac(v)
                w = model(w, U)

                cost = qcd_ml.util.linear_algebra.innerproduct(w-v, w-v).real

                approx_history.append(cost.item())

    return (np.mean(approx_history), np.std(approx_history))


def plotting():
        
    with torch.no_grad():

        plt.figure(figsize=(8, 5), dpi=100)
        plt.title("Approximation Quality over training steps")
        plt.plot(range(len(loss_history)), loss_history, label = "Approximation Quality")
        plt.yscale('log')
        plt.xlabel("Training Steps")
        plt.ylabel("Approximation Quality")
        plt.legend()
        plt.grid(True, which="both", axis="both", alpha=0.3, ls="--")
        plt.tight_layout()
        plt.show()
        theme.apply_transforms()



def calc_m(Operator, NeuralNetwork, layers, pathtype: int):

    m_range = (config.m,)
    print(Operator, "- Using", NeuralNetwork.__name__, layers, "layers", "with", config.training_steps, "steps and pathtype", pathtype)
    print("mass,test_mean,test_std")
    load_data()

    for m in m_range:

        create_dirac_operator(Operator, m, config.csw)

        model = NeuralNetwork(path_generator(pathtype), layers).to(device)
        model.init_identity(noise = 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))

        training_loop_lowfeedback(config, model, optimizer, config.training_steps)
        #training_loop_auto(config, model, optimizer, config.treshhold, config.max_training_steps)

        approx_mean, approx_std = testing_loop(config, model, config.testing_sample)

        print(f"{m},{approx_mean},{approx_std}")


    print("end")



def run(Operator, NeuralNetwork, layers, pathtype: int):

    print(Operator, "- Using", NeuralNetwork.__name__, layers, "layers", "with", config.training_steps, "steps and pathtype", pathtype)
    load_data()
    create_dirac_operator(Operator, config.m, config.csw)
    print("Mass =", config.m)
    model = NeuralNetwork(path_generator(pathtype), layers).to(device)
    
    model.init_identity(noise = 0.001)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))

    training_loop(config, model, optimizer, config.training_steps)

    approx_mean, approx_std = testing_loop(config, model, config.testing_sample)

    print("Mean Approx:", approx_mean)
    print("Std Approx:", approx_std)

    plotting()

    print("end")



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--mass", type=float, help="Mass parameter of dirac operator", default=None)
    parser.add_argument("--task-id", type=int, help="Slurm array task id", default=None)
    args = parser.parse_args()

    m_range = np.arange(-5, 2.5, 0.5)

    if args.mass is not None:
        config.m = args.mass
        
    elif args.task_id is not None:
        config.m = float(m_range[args.task_id])





    run(Operator="DW", NeuralNetwork = models.HCRes_VarAdapt, layers = 1, pathtype = 2)
    #calc_m(Operator="DW", NeuralNetwork = models.HCRes_VarAdapt, layers = 1,  pathtype = 2)

    #HL1
    #HL2
    #HL1_Res1
    #HL1_Res2
    #HC_Var
    #HCRes_Var
    #HCRes_VarAdapt
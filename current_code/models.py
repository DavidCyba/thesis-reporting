import qcd_ml
import torch
import clifford

# Linear-PT Model

class HL1(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()

        self.paths = paths

        self.L1 = qcd_ml.nn.dense.v_Dense(n_feature_in= 1, n_feature_out= 9)
        self.L2 = qcd_ml.nn.dense.v_Dense(n_feature_in= 9, n_feature_out= 1)

    def forward(self, x, U):


        # instance PT in forward to keep same model for different U
        PT1 = qcd_ml.nn.pt.v_PT(self.paths, U)

        x = x.unsqueeze(0)

        x = self.L1(x)
        x = PT1(x)
        x = self.L2(x)


        x = x.squeeze(0)

        return x



    def init_identity(self, noise = 0.0):
        with torch.no_grad():

            I = torch.eye(
                4,
                dtype=self.L1.weights.dtype,
                device=self.L1.weights.device
            )

            self.L1.weights[0, 0].copy_(I)

            self.L2.weights.zero_()
            self.L2.weights[0,0].copy_(I)

            if noise > 0:
                for layer in (self.L1, self.L2):
                    layer.weights.add_(
                        noise * torch.randn_like(layer.weights)
                    )

    def init_hopping(self, m, noise = 0.0):
        kappa = 1/(2*(m+4))
        #to be finished


class HL1_Res1(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()

        self.paths = paths

        self.L1 = qcd_ml.nn.dense.v_Dense(n_feature_in= 1, n_feature_out= 9)
        self.L2 = qcd_ml.nn.dense.v_Dense(n_feature_in= 9, n_feature_out= 1)

    def forward(self, x, U):


        # instance PT in forward to keep same model for different U
        PT1 = qcd_ml.nn.pt.v_PT(self.paths, U)

        x = x.unsqueeze(0)

        res = x

        x = self.L1(x)
        x = PT1(x)
        x = self.L2(x)

        x = res + x


        x = x.squeeze(0)

        return x



    def init_identity(self, noise = 0.0):
        with torch.no_grad():

            I = torch.eye(
                4,
                dtype=self.L1.weights.dtype,
                device=self.L1.weights.device
            )

            self.L1.weights[0, 0].copy_(I)

            self.L2.weights.zero_()

            if noise > 0:
                for layer in (self.L1, self.L2):
                    layer.weights.add_(
                        noise * torch.randn_like(layer.weights)
                    )





class HL1_Res2(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()

        self.paths = paths

        self.L1 = qcd_ml.nn.dense.v_Dense(n_feature_in= 1, n_feature_out= 9)
        self.L2 = qcd_ml.nn.dense.v_Dense(n_feature_in= 9, n_feature_out= 1)

    def forward(self, x, U):


        # instance PT in forward to keep same model for different U
        PT1 = qcd_ml.nn.pt.v_PT(self.paths, U)

        h = self.L1(x.unsqueeze(0))
        h = PT1(h)

        h = h.clone()
        h[0] = h[0] + x #Adds x to first feature channel corresponding to PT [[]]

        output = self.L2(h).squeeze(0)

        return output + h[0]



    def init_identity(self, noise = 0.0):
        with torch.no_grad():

            I = torch.eye(
                4,
                dtype=self.L1.weights.dtype,
                device=self.L1.weights.device
            )

            self.L1.weights[0].copy_(I)

            self.L1.weights[0,0].zero_()

            self.L2.weights.zero_()

            if noise > 0:
                for layer in (self.L1, self.L2):
                    layer.weights.add_(
                        noise * torch.randn_like(layer.weights)
                    )

class HL2(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()

        self.paths = paths

        self.L1 = qcd_ml.nn.dense.v_Dense(n_feature_in= 1, n_feature_out= 9)
        self.L3 = qcd_ml.nn.dense.v_Dense(n_feature_in= 9, n_feature_out= 1)
        self.L2 = qcd_ml.nn.dense.v_Dense(n_feature_in= 9, n_feature_out= 9)

    def forward(self, x, U):


        # instance PT in forward to keep same model for different U
        PT = qcd_ml.nn.pt.v_PT(self.paths, U)

        x = x.unsqueeze(0)

        x = self.L1(x)
        x = PT(x)
        x = self.L2(x)
        x = PT(x)
        x = self.L3(x)

        x = x.squeeze(0)

        return x

    def init_identity(self, noise = 0.0):
        with torch.no_grad():

            I = torch.eye(
                4,
                dtype=self.L1.weights.dtype,
                device=self.L1.weights.device
            )

            self.L1.weights[0].copy_(I)

            self.L2.weights.zero_()
            self.L2.weights[0].copy_(I)

            self.L3.weights.zero_()
            self.L3.weights[0,0].copy_(I)

            if noise > 0:
                for layer in (self.L1, self.L2, self.L3):
                    layer.weights.add_(
                        noise * torch.randn_like(layer.weights)
                    )        


class HL3(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()

        self.paths = paths

        self.L1 = qcd_ml.nn.dense.v_Dense(n_feature_in= 1, n_feature_out= 9)
        self.L3 = qcd_ml.nn.dense.v_Dense(n_feature_in= 9, n_feature_out= 9)
        self.L2 = qcd_ml.nn.dense.v_Dense(n_feature_in= 9, n_feature_out= 9)
        self.L4 = qcd_ml.nn.dense.v_Dense(n_feature_in= 9, n_feature_out= 1)


    def forward(self, x, U):


        # instance PT in forward to keep same model for different U
        PT = qcd_ml.nn.pt.v_PT(self.paths, U)

        x = x.unsqueeze(0)

        x = self.L1(x)
        x = PT(x)
        x = self.L2(x)
        x = PT(x)
        x = self.L3(x)
        x = PT(x)
        x = self.L4(x)

        x = x.squeeze(0)

        return x

    def init_identity(self, noise = 0.0):
        with torch.no_grad():

            I = torch.eye(
                4,
                dtype=self.L1.weights.dtype,
                device=self.L1.weights.device
            )

            self.L1.weights[0].copy_(I)

            self.L2.weights.zero_()
            self.L2.weights[0].copy_(I)

            self.L3.weights.zero_()
            self.L3.weights[0,0].copy_(I)

            self.L4.weights.zero_()
            self.L4.weights[0,0].copy_(I)

            if noise > 0:
                for layer in (self.L1, self.L2, self.L3, self.L4):
                    layer.weights.add_(
                        noise * torch.randn_like(layer.weights)
                    )        


# HC network

class HC1(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()

        self.paths = paths

        self.C1 = clifford.Clifford(n_feature_in= 1, n_feature_out= 9)
        self.C2 = clifford.Clifford(n_feature_in= 9, n_feature_out= 1)

    def forward(self, x, U):


        # instance PT in forward to keep same model for different U
        PT1 = qcd_ml.nn.pt.v_PT(self.paths, U)

        x = x.unsqueeze(0)

        x = self.C1(x)
        x = PT1(x)
        x = self.C2(x)


        x = x.squeeze(0)

        return x

    def init_identity(self, noise = 0.0):
        with torch.no_grad():

            self.C1.weights.zero_()
            self.C2.weights.zero_()

            self.C1.weights[0,0,0]= 1.0
            self.C2.weights[0,0,0]= 1.0

            if noise > 0:
                for layer in (self.C1, self.C2):
                    layer.weights.add_(
                        noise * torch.randn_like(layer.weights)
                    )


# Clifford-PT variable layers

class HC_Var(torch.nn.Module):
    def __init__(self, paths, n_layers = 1):
        super().__init__()

        self.paths = paths

        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        self.C1 = clifford.Clifford(n_feature_in= 1, n_feature_out= 9)
        self.C2 = clifford.Clifford(n_feature_in= 9, n_feature_out= 1)

        self.layers = torch.nn.ModuleList([
            clifford.Clifford(n_feature_in= 9, n_feature_out= 9)
            for _ in range(n_layers - 1)
        ])

    def forward(self, x, U):


        # instance PT in forward to keep same model for different U
        PT = qcd_ml.nn.pt.v_PT(self.paths, U)

        x = x.unsqueeze(0)

        x = self.C1(x)

        x = PT(x)

        for layer in self.layers:
            
            x = layer(x)
            x = PT(x)
        
        x = self.C2(x)


        x = x.squeeze(0)

        return x

    def init_identity(self, noise = 0.0):
        with torch.no_grad():

            all_layers = (self.C1, *self.layers, self.C2)

            for layer in all_layers:
                layer.weights.zero_()


            self.C1.weights[0,0,0]= 1.0
            self.C2.weights[0,0,0]= 1.0

            for layer in self.layers:
                layer.weights[0,0,0] = 1.0

            if noise > 0:
                for layer in all_layers:
                    layer.weights.add_(
                        noise * torch.randn_like(layer.weights)
                    )


# Clifford-PT residual with variable layers

class HCRes_Var(torch.nn.Module):
    def __init__(self, paths, n_layers = 1):
        super().__init__()

        self.paths = paths

        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        self.C1 = clifford.Clifford(n_feature_in= 1, n_feature_out= 9)
        self.C2 = clifford.Clifford(n_feature_in= 9, n_feature_out= 1)

        self.layers = torch.nn.ModuleList([
            clifford.Clifford(n_feature_in= 9, n_feature_out= 9)
            for _ in range(n_layers - 1)
        ])

    def forward(self, x, U):


        PT = qcd_ml.nn.pt.v_PT(self.paths, U)

        h = self.C1(x.unsqueeze(0))

        h = PT(h)

        h = h.clone()
        h[0] = h[0] + x

        for layer in self.layers:

            h = h + PT(layer(h))

        output = self.C2(h).squeeze(0)

        return output + x

    def init_identity(self, noise = 0.0):
        with torch.no_grad():

            all_layers = (self.C1, *self.layers, self.C2)

            for layer in all_layers:
                layer.weights.zero_()

                if noise > 0:
                    layer.weights.add_(
                        noise * torch.randn_like(layer.weights)
                    )



# Clifford-PT residual with variable layers and adaptive in/out features

class HCRes_VarAdapt(torch.nn.Module):
    def __init__(self, paths, n_layers = 1):
        super().__init__()

        self.paths = paths

        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        self.C1 = clifford.Clifford(n_feature_in= 1, n_feature_out= len(paths))
        self.C2 = clifford.Clifford(n_feature_in= len(paths), n_feature_out= 1)

        self.layers = torch.nn.ModuleList([
            clifford.Clifford(n_feature_in= len(paths), n_feature_out= len(paths))
            for _ in range(n_layers - 1)
        ])

    def forward(self, x, U):


        PT = qcd_ml.nn.pt.v_PT(self.paths, U)

        h = self.C1(x.unsqueeze(0))

        h = PT(h)

        h = h.clone()
        h[0] = h[0] + x

        for layer in self.layers:
            residual = h[0].clone()

            h = PT(layer(h))

            h = h.clone()
            h[0] = h[0] + residual

        output = self.C2(h).squeeze(0)

        return output + h[0]

    def init_identity(self, noise = 0.0):
        with torch.no_grad():

            all_layers = (self.C1, *self.layers, self.C2)

            for layer in all_layers:
                layer.weights.zero_()

                if noise > 0:
                    layer.weights.add_(
                        noise * torch.randn_like(layer.weights)
                    )
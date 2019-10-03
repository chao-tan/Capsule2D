from torch import nn
import capsnn

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()

        model = [nn.Conv2d(in_channels=1,out_channels=16,kernel_size=1,padding=0,stride=1),
                 nn.BatchNorm2d(16),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 16
        out_features = in_features*2
        for i in range(3):
            model += [  nn.Conv2d(in_channels=in_features,out_channels=out_features,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(out_features),
                        nn.ReLU(inplace=True)]

            if i != 2:
                model += [ nn.Conv2d(in_channels=out_features,out_channels=out_features,kernel_size=3,stride=2,padding=1),
                           nn.BatchNorm2d(out_features),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            else:
                model += [nn.Conv2d(in_channels=out_features,out_channels=out_features,kernel_size=3,stride=1,padding=1),
                          nn.BatchNorm2d(out_features),
                          nn.ReLU(inplace=True),
                          nn.BatchNorm2d(out_features),
                          nn.ReLU(inplace=True)]

            in_features = out_features
            out_features = in_features*2

        # Upsampling
        out_features = in_features//2
        for i in range(3):
            model += [  nn.ConvTranspose2d(in_channels=in_features,out_channels=in_features,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(in_features),
                        nn.ReLU(inplace=True) ]

            if i != 2:
                model += [nn.ConvTranspose2d(in_channels=in_features,out_channels=out_features,kernel_size=3,stride=2,padding=1,output_padding=1),
                          nn.BatchNorm2d(out_features),
                          nn.ReLU(inplace=True)]
            else:
                model += [nn.ConvTranspose2d(in_channels=in_features,out_channels=out_features,kernel_size=3,stride=1,padding=1),
                          nn.BatchNorm2d(out_features),
                          nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # # Output layer
        model += [nn.Conv2d(in_channels=16,out_channels=1,kernel_size=1,padding=0,stride=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class CapsBasicNet_PART1(nn.Module):
    def __init__(self):
        super(CapsBasicNet_PART1, self).__init__()

        model = [nn.Conv2d(in_channels=1,out_channels=16,kernel_size=1,padding=0,stride=1),
                 nn.BatchNorm2d(16),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 16
        out_features = in_features*2
        for i in range(3):
            model += [  nn.Conv2d(in_channels=in_features,out_channels=out_features,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(out_features),
                        nn.ReLU(inplace=True)]

            if i != 2:
                model += [ nn.Conv2d(in_channels=out_features,out_channels=out_features,kernel_size=3,stride=2,padding=1),
                           nn.BatchNorm2d(out_features),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            else:
                model += [nn.Conv2d(in_channels=out_features,out_channels=out_features,kernel_size=3,stride=1,padding=1),
                          nn.BatchNorm2d(out_features),
                          nn.ReLU(inplace=True),
                          nn.BatchNorm2d(out_features),
                          nn.ReLU(inplace=True)]

            in_features = out_features
            out_features = in_features*2

        # Capsule transform
        model += [capsnn.Conv2CapsuleConv2D(in_channels=128,out_channels=128,dim_caps=8,kernel_size=1,stride=1,padding=0),
                  capsnn.caps_Conv2d(in_channels=16,out_channels=16,in_capsdim=8,out_capsdim=8,kernel_size=3,padding=1,stride=1,routing_nums=3)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)





class CapsBasicNet_PART2(nn.Module):
    def __init__(self):
        super(CapsBasicNet_PART2,self).__init__()

        in_features = 128
        out_features = in_features // 2

        model = [capsnn.CapsuleConv2D2Conv(in_channels=in_features,out_channels=in_features,kernel_size=1,padding=0,stride=1),
                 nn.BatchNorm2d(in_features),
                 nn.ReLU()]

        # Upsampling
        for i in range(3):
            model += [nn.ConvTranspose2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1,
                                         padding=1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(inplace=True)]

            if i != 2:
                model += [
                    nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    nn.BatchNorm2d(out_features),
                    nn.ReLU(inplace=True)]
            else:
                model += [
                    nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1,
                                       padding=1),
                    nn.BatchNorm2d(out_features),
                    nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, padding=0, stride=1)]


        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class CapsBasicNet(nn.Module):
    def __init__(self):
        super(CapsBasicNet,self).__init__()
        self.part1 = CapsBasicNet_PART1()
        self.part2 = CapsBasicNet_PART2()

    def forward(self, x):
        out_caps = self.part1(x)
        out = self.part2(out_caps)

        return out,out_caps

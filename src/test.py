import hayati_model
import json
import numpy as np

def main():
    with open("src/config/ARM95_calibration.json", 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    model.nominal_dh = np.array([
    [0,1.5708,0,3.14,0],
    [0.46, 0, 0,1.5708,1],
    [ 0.38, 0, 0, 0, 1],
    [0,-1.5708,-0.135, -1.5708,0],
    [0,-1.5708,0.109,0,0],
    [0,0,0.176,0, 0]
    ])
    model.estimated_dh = np.array([
        [-0.0012779183803429634, 1.572714450821018, -2.5062639459755162e-05, 3.1422950951906374, 0.0],
        [0.4574230096363143, 0.0, -0.004532849759717515, 1.5662811768078075, 1.0],
        [0.37935629619888683, 0.0003489506736117725, 0.021955132250431407, -0.0038254714558886276, 1.0],
        [-0.00016091350665088067, -1.570828117172346, -0.13788202710337846, -1.5708332140610803, 0.0],
        [-0.0009972816983362385, -1.5688969089496632, 0.11017597889436716, 0.0, 0.0],
        [-4.681571669389498e-06, 0.0, 0.16736879843437474, 1.2752111549738254e-08, 0.0]
    ])
    
    nom_params = model.get_readable_params("nominal")
    est_params = model.get_readable_params("estimated")
    print(nom_params[:, :4])
    print()
    print(est_params[:, :4])

if __name__ == "__main__":
    main()

from pyroll.freiberg_flow_stress import FreibergFlowStressCoefficients

Steel_54SiCr6 = {
    "material": "54SiCr6",
    "density": 7.7e3,
    "specific_heat_capacity": 490,
    "thermal_conductivity": 47,
    "elastic_modulus": 120000.0e6,
    "poissons_ratio": 0.3,
    "freiberg_flow_stress_coefficients": FreibergFlowStressCoefficients(
         a=759.285 * 1e6,
         m1=-0.00358,
         m2=0.338339,
         m3=-0.055943,
         m4=0.000104,
         m5=-0.000599,
         m6=0,
         m7=-0.448028,
         m8=0.000198,
         m9=0.356106,
         baseStrain=0.1,
         baseStrainRate=0.1),
}

Steel_100Cr6 = {
    "material": "100Cr6",
    "density": 7.3719e3,  # @1000 °C
    "specific_heat_capacity": 695,  # @ 1000 °C
    "thermal_conductivity": 28.25,  # @ 1000 °C
    "elastic_modulus": 81700.0e6,  # @ 1000 °C
    "poissons_ratio": 0.35,  # @ 1000 °C
    "freiberg_flow_stress_coefficients": FreibergFlowStressCoefficients(
        a=8422.43 * 1e6,
        m1=-0.00361,
        m2=0.36013,
        m3=0,
        m4=0.00367,
        m5=-0.00160,
        m6=0,
        m7=0.32298,
        m8=0.000152,
        m9=0,
        baseStrain=0.1,
        baseStrainRate=0.1,),
}

Steel_11SMn30 = {
    "material": "11SMn30",
    "density": 7.5e3,
    "specific_heat_capacity": 690,
    "thermal_conductivity": 23,
    "elastic_modulus": 120000.0e6,
    "poissons_ratio": 0.29,
    "freiberg_flow_stress_coefficients": FreibergFlowStressCoefficients(
        a=1473.43 * 1e6,
        m1=-0.00239,
        m2=0.22924,
        m3=0.0,
        m4=-0.00870,
        m5=0.00010,
        m6=0,
        m7=-0.44065,
        m8=0.000161,
        m9=0,
        baseStrain=0.01,
        baseStrainRate=0.1),
}


Steel_C45 = {
    'material': 'C45',
    'density': 7.85e3,
    'specific_heat_capacity': 645,
    'thermal_conductivity': 23,
    'elastic_modulus': 120000.e+6,
    'poissons_ratio': 0.3,
    'freiberg_flow_stress_coefficients': FreibergFlowStressCoefficients(
        a=3268.49 * 1e6,
        m1=-0.00267855,
        m2=0.34446,
        m3=0.,
        m4=0.000551814,
        m5=-0.00132042,
        m6=0,
        m7=0.0166334,
        m8=0.000149907,
        m9=0,
        baseStrain=0.1,
        baseStrainRate=0.1),
}

Steel_BST500 = {
    'material': 'BST500',
    'density': 7.5e3,
    'specific_heat_capacity': 690,
    'thermal_conductivity': 23,
    'elastic_modulus': 120000.e+6,
    'poissons_ratio': 0.3,
    'freiberg_flow_stress_coefficients': FreibergFlowStressCoefficients(
        a=4877.12 * 1e6,
        m1=-0.00273339,
        m2=0.302309,
        m3=-0.0407581,
        m4=0.000222222,
        m5=-0.000383134,
        m6=0,
        m7=-0.492672,
        m8=0.000175044,
        m9=-0.0611783,
        baseStrain=0.1,
        baseStrainRate=0.1),
}

Steel_C56D = {
    'material': 'C56D',
    'density': 7.4e3,
    'specific_heat_capacity': 690,
    'thermal_conductivity': 27,
    'elastic_modulus': 120000.e+6,
    'poissons_ratio': 0.3,
    'freiberg_flow_stress_coefficients': FreibergFlowStressCoefficients(
        a=3006.43 * 1e6,
        m1=-0.00324,
        m2=0.03674,
        m3=0,
        m4=-0.02247,
        m5=-0.00021,
        m6=0,
        m7=-0.24543,
        m8=0.000159,
        m9=0,
        baseStrain=0.1,
        baseStrainRate=0.1),
}
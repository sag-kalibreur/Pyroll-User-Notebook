Temperature = {'title': 'Temperature', 'unit': '°C', 'fmt': '{:.0f}', 'label': 'center', 'font_color': 'white'}
strain = {'title': 'Strain', 'unit': '-', 'fmt': '{:.2f}', 'label': 'center', 'font_color': 'white'}
strain_rate = {'title': 'Strainrate', 'unit': '1/s', 'fmt': '{:.2f}', 'label': 'center', 'font_color': 'white'}
velocity = {'title': 'Rolling speed', 'unit': 'm/s', 'fmt': '{:.2f}', 'label': 'center', 'font_color': 'white'}
stress = {'title': 'Stress', 'unit': 'MPa', 'fmt': '{:.1f}', 'label': 'center', 'font_color': 'white'}
percent = {'title': '-', 'unit': '%', 'fmt': '{:.1f}', 'label': 'center', 'font_color': 'white'}
time = {'title': 'Time', 'unit': 's', 'fmt': '{:.2f}', 'label': 'center', 'font_color': 'white'}
frequency = {'title': 'Frequency', 'unit': '1/min', 'fmt': '{:.2f}', 'label': 'center', 'font_color': 'white'}
unitless = {'title': '-', 'unit': '-', 'fmt': '{:.2f}', 'label': 'center', 'font_color': 'white'}
mass_flux = {'title': 'Mass flux', 'unit': 'kg/s', 'fmt': '{:.2f}', 'label': 'center', 'font_color': 'white'}
radius = {'title': 'Radius', 'unit': 'mm', 'fmt': '{:.1f}', 'label': 'center', 'font_color': 'white'}
diameter = {'title': 'Diameter', 'unit': 'mm', 'fmt': '{:.1f}', 'label': 'center', 'font_color': 'white'}
length = {'title': 'Length', 'unit': 'mm', 'fmt': '{:.1f}', 'label': 'center', 'font_color': 'white'}
ratio = {'title': 'Ratio', 'unit': 'mm', 'fmt': '{:.3f}', 'label': 'center', 'font_color': 'white'}

annotations = {
             'in_profile_temperature' : Temperature,
	     'in_profile_surface_temperature' : Temperature,
	     'in_profile_core_temperature' : Temperature,
	     'out_profile_surface_temperature' : Temperature,
	     'out_profile_core_temperature' : Temperature,
             'strain': strain,
             'strain_rate': strain_rate,
             'roll_working_velocity': velocity,
             'in_profile_velocity': velocity,
             'out_profile_velocity': velocity,
             'deformation_resistance': stress,
             'reduction': percent,
             'duration': time,
             'velocity': velocity,
             'roll_rotational_frequency': frequency,
             'spread': unitless,
             'elongation': unitless,
             'area_ratio': unitless,
             'mass_flux': mass_flux,
             'roll_working_radius': radius,
            'roll_nominal_radius': radius,
             'roll_working_diameter': diameter,
            'roll_nominal_diameter': diameter,
            'out_profile_filling_ratio': unitless,
            'out_profile_cross_section_height': length,
            'DT': length,
            'freq_ratio': ratio,
             'velo_ratio': ratio,
             }
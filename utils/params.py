
# Month names
month_names=['Aug_p','Sep_p','Oct_p','Nov_p','Dec_p',
             'Jan','Feb','Mar','Apr','May','Jun','Jul',
             'Aug','Sep','Oct','Nov','Dec']
month_names_no_prev_year=['Jan','Feb','Mar','Apr','May',
                          'Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_names_with_drivers=[f'{mode}-{name}' for mode in ['PPT','AT','PAR'] for name in month_names]
month_names_with_drivers_no_prev_year=[f'{mode}-{name}' for mode in ['PPT','AT','PAR'] for name in month_names_no_prev_year]

# Feature groups for coloring schemes of the graphs:

# Seasons:
season_groups=[['Aug_p','Sep_p','Oct_p','Nov_p','Dec_p'],
               ['Jan','Feb'],
               ['Mar','Apr','May'],
               ['Jun','Jul','Aug'],['Sep','Oct','Nov','Dec']]
season_groups_no_prev_year=[['Jan','Feb'],
                            ['Mar','Apr','May'],
                            ['Jun','Jul','Aug'],
                            ['Sep','Oct','Nov','Dec']]
seasons_groups_with_drivers=[ [f'{mode}-{name}' for mode in ['PPT','AT','PAR'] for name in season] for season in season_groups]
seasons_groups_with_drivers_no_prev_year=[ [f'{mode}-{name}' for mode in ['PPT','AT','PAR'] for name in season] for season in season_groups_no_prev_year]

# Features:
features_groups_with_drivers=[ [ f'{mode}-{name}' for name in month_names] for mode in ['PPT','AT','PAR']]
features_groups_with_drivers_no_prev_year=[ [ f'{mode}-{name}' for name in month_names_no_prev_year] for mode in ['PPT','AT','PAR']]
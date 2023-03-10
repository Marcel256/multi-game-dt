import os





games = [
'Pong',
'Qbert',
'DemonAttack',
'SpaceInvaders',
'Breakout',
]
file_types = ['observation', 'action', 'reward', 'terminal']
file_range = range(41, 43)

def get_file_name(type, file_number):
    return '$store$_{}_ckpt.{}.gz'.format(type, file_number)

url_pattern = 'http://storage.googleapis.com/atari-replay-datasets/dqn/{}/1/replay_logs/{}'
download_dir = 'data/download'

for game in games:
    game_dir = os.path.join(download_dir, game)
    if not os.path.exists(game_dir):
        os.makedirs(game_dir)
    for i in file_range:
        for file_type in file_types:
            file_name = get_file_name(file_type, i)
            path = url_pattern.format(game, file_name)
            print('Downloading ', path)
            out_file = os.path.join(game_dir, file_name)
            os.system(f"""wget -O "{out_file}" "{path}""")

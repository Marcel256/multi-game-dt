import os
from subprocess import Popen







games = ['Asterix',
'BeamRider',
'Breakout',
'DemonAttack',
'Gravitar',
'TimePilot',
'SpaceInvaders',
'Jamesbond',
'Assault',
'Frostbite']
file_types = ['observation', 'action', 'reward', 'terminal']
file_range = range(50, 51)

def get_file_name(type, file_number):
    return '$store$_{}_ckpt.{}.gz'.format(type, file_number)

uri_pattern = 'gs://atari-replay-datasets/dqn/{}/1/replay_logs/{}'
download_dir = 'data/download'

for game in ['Asterix']:
    game_dir = os.path.join(download_dir, game)
    if not os.path.exists(game_dir):
        os.makedirs(game_dir)
    for i in file_range:
        for file_type in file_types:
            file_name = get_file_name(file_type, i)
            path = uri_pattern.format(game, file_name)
            print('Downloading ', path)
            out_file = os.path.join(game_dir, file_name)
            p = Popen(['gsutil', '-m', 'cp', '-R', path, out_file])
            p.wait()



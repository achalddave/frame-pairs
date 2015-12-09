"""Generate an HTML viewer for frame pairs."""

import argparse
import json

import jinja2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('frame_pairs_directory',
                    default=None,
                    help='Directory containing output of dump_frame_pairs.py')
args = parser.parse_args()


def main():
    jinja_env = jinja2.Environment(
        loader=jinja2.PackageLoader(__name__, 'templates'))
    frame_pairs_template = jinja_env.get_template('frame_pairs.html')
    with open('{}/output.json'.format(args.frame_pairs_directory)) as f:
        frame_pairs_info = json.load(f)
    with open('{}/viewer.html'.format(args.frame_pairs_directory), 'wb') as f:
        f.write(frame_pairs_template.render(video_frame_pairs=
                                            frame_pairs_info))

if __name__ == '__main__':
    main()

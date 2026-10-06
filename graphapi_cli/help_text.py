"""Plain-language command descriptions and copyable examples."""
import argparse


COMMAND_HELP = {
    '': (
        'Run GRAPH-API in Docker: Habitat scenes, TIAGO robots and recorded TIAGO bags.\n'
        'Use COMMAND --help to see its options and examples.\n'
        'Replace paths with your own paths and RUN_ID with an ID from ./graphapi status.',
        [('Prepare Habitat', 'setup sim --dataset /path/to/hm3d'),
         ('Run a Habitat scene in the background', 'run sim --scene hm3d_00861 --detach'),
         ('Run a recorded TIAGO bag (requires private PAL Docker)', 'run tiago bag /data/bags/example --no-record --detach'),
         ('Open the live dashboard', 'dashboard --mode live'),
         ('Find a run and stop it', 'status', 'stop RUN_ID')]),
    'init': ('Create local settings. Keep settings that already exist.',
             [('Create config/local.yaml', 'init'), ('Choose where to save run files', 'init --workspace /data/graphapi')]),
    'setup': ('Prepare Docker, dependencies and data paths for a run.\n'
              'Missing public dependencies and model files may be installed or downloaded.\n'
              'TIAGO needs your private PAL container or private build files.',
              [('Prepare Habitat', 'setup sim --dataset /path/to/hm3d'),
               ('Choose model storage and the workspace', 'setup sim --dataset /path/to/hm3d --models /data/models --workspace /data/graphapi'),
               ('Register private PAL build files', 'setup tiago --pal-bundle /private/TIAGO_ISO'),
               ('Use a prepared private PAL container', 'setup tiago --container tiago-127-dev')]),
    'run': ('Start a Habitat scene, a physical TIAGO robot or a recorded TIAGO bag.\n'
            'Perception and recording are on by default. Use --detach to run in the background.\n'
            "Export REGOLO_API_KEY='your-key' before starting. A missing VLM key stops startup.",
            [('Habitat', 'run sim --scene hm3d_00861 --detach'),
             ('Physical TIAGO', 'run tiago physical'),
             ('Recorded TIAGO bag', 'run tiago bag /data/bags/example --no-record --detach')]),
    'run sim': ('Run a Habitat scene. By default, use the simulator position and run without windows.\n'
                'You do not need --profile sim-gt: that is the default.\n'
                'Run setup sim first. Full perception needs REGOLO_API_KEY in your terminal.',
                [('Run one scene in the background', 'run sim --scene hm3d_00861 --one-storey --detach'),
                 ('Open visualization windows without saving another recording', 'run sim --scene hm3d_00861 --gui --no-record'),
                 ('Choose one floor by height in metres', 'run sim --scene hm3d_00861 --floor=-1.59'),
                 ('Build a map without object detection', 'run sim --scene hm3d_00861 --mapping-only')]),
    'run tiago': ('Choose how to run TIAGO. Both choices require your private PAL Docker.\n'
                  'physical connects to a robot. bag replays a recording without a robot.',
                  [('Physical robot', 'run tiago physical --detach'),
                   ('Recorded bag', 'run tiago bag /data/bags/example --no-record --detach')]),
    'run tiago physical': ('Connect to a physical TIAGO robot using private PAL Docker.\n'
                           'RTAB-Map builds the application map by default.\n'
                           'Perception and recording are on. Full perception needs REGOLO_API_KEY.',
                           [('Start the robot pipeline in the background', 'run tiago physical --detach'),
                            ('Show RViz', 'run tiago physical --gui'),
                            ('Check cameras and mapping without object detection', 'run tiago physical --no-perception --no-record'),
                            ('Reuse the current TIAGO session and output', 'run tiago physical --container tiago-127-dev --resume')]),
    'run tiago bag': ('Replay a recorded TIAGO bag using private PAL Docker. No robot connection is needed.\n'
                      'Use the map and transforms saved in the bag by default.\n'
                      'Perception and recording are on. Full perception needs REGOLO_API_KEY.\n'
                      'In the dashboard, click ANNOTATIONS to keep the last completed detection visible.',
                      [('Test perception without saving another recording', 'run tiago bag /data/bags/example --no-record --detach'),
                       ('Replay repeatedly until you stop the run', 'run tiago bag /data/bags/example --loop --no-record --detach'),
                       ('Check cameras and the map without object detection', 'run tiago bag /data/bags/example --no-perception --no-record'),
                       ('Build a new map with RTAB-Map', 'run tiago bag /data/bags/example --map-source slam'),
                       ('Play at half speed', 'run tiago bag /data/bags/example --rate 0.5')]),
    'doctor': ('Check Docker, settings and the files needed for your chosen run.\n'
               'Use --live for extra checks inside the private TIAGO container.',
               [('Check Habitat', 'doctor --mode sim'),
                ('Check TIAGO bag replay', 'doctor --mode tiago-bag --live'),
                ('Check physical TIAGO', 'doctor --mode tiago-physical --live')]),
    'batch': ('Run the experiments listed in a YAML schedule, one after another.\n'
              'Skip experiments that already completed unless you use --force.',
              [('See the planned experiments without running them', 'batch config/example.runs.yaml --dry-run'),
               ('Run the schedule', 'batch config/full.runs.yaml'),
               ('Continue after a failed experiment', 'batch config/full.runs.yaml --continue-on-failure')]),
    'status': ('Show running or starting runs and dashboards. Use --all to include stopped runs.\n'
               'Copy an ID from this list to use with logs, stop or view.',
               [('Show active runs and dashboards', 'status'),
                ('Show one run', 'status RUN_ID'), ('Show past runs too', 'status --all'),
                ('Get JSON output', 'status --json')]),
    'logs': ('Read the startup log for a run. Use --follow to see new lines as they arrive.\n'
             'If more than one run or dashboard is active, give its ID.',
             [('Read a run log', 'logs RUN_ID'), ('Watch new log lines', 'logs RUN_ID --follow')]),
    'attach': ('Watch a run log until the run stops. Press Ctrl+C to stop watching.\n'
               'This does not open a shell or stop the run.',
               [('Watch a run', 'attach RUN_ID')]),
    'stop': ('Stop a run or dashboard and let it finish saving its files.\n'
             'Use its ID when several are active. Stopping a run leaves its dashboard open.\n'
             'DRAINING means shutdown is in progress; check status until it finishes.',
             [('Find the ID, then stop that run', 'status', 'stop RUN_ID'),
              ('Check that shutdown finished', 'status RUN_ID'),
              ('Stop the only active run or dashboard', 'stop active')]),
    'dashboard': ('Open the web dashboard. The default address is http://127.0.0.1:8082.\n'
                  'live shows the running pipeline; replay shows saved results.\n'
                  'Click ANNOTATIONS to see the last completed annotated image in live mode.',
                  [('Show a running pipeline', 'dashboard --mode live'),
                   ('Show the last completed run', 'dashboard latest --mode replay'),
                   ('Show a specific saved run', 'dashboard RUN_ID --mode replay'),
                   ('Use another port', 'dashboard --mode live --port 8083')]),
    'view': ('Open RViz on this computer for a running pipeline.\n'
             'If several pipelines are running, give the ID from status.',
             [('Open RViz for the only running pipeline', 'view'),
              ('Open RViz for a specific run', 'view RUN_ID')]),
    'eval': ('Evaluate saved run results inside Docker. Write reports into the run folder.\n'
             'latest means the last completed run, not a run still in progress.',
             [('Evaluate the last completed run', 'eval latest'),
              ('Evaluate a specific run', 'eval RUN_ID'),
              ('Rebuild the ground-truth data used for evaluation', 'eval RUN_ID --force')]),
    'tools': ('Find and run the other tools included in this repository.',
              [('List available tools', 'tools list'),
               ('Show tests and internal tools too', 'tools list --all'),
               ('Preview a tool command', 'tools run --dry-run launch-simulation')]),
    'tools list': ('List tool names and where they run.',
                   [('List user-facing tools', 'tools list'), ('Include tests and internal tools', 'tools list --all')]),
    'tools run': ('Run a tool by its name from tools list.\n'
                  'Put tool options after -- to pass them to the tool.',
                  [('Preview a launch command', 'tools run --dry-run launch-simulation -- map_yaml:=/data/maps/office.yaml'),
                   ('Ask a tool for its own help', 'tools run schedule-runs -- --help')]),
    'launch': ('Start a TIAGO ROS launch file. Gazebo uses the public image; bag launches need private PAL Docker.\n'
               'Use run tiago bag for the main recorded-bag pipeline.',
               [('Start TIAGO in Gazebo', 'launch tiago-gazebo -- map_yaml:=/data/maps/office.yaml'),
                ('Show arguments for a bag launch', 'launch tiago-bag-slam --help'),
                ('Replay a bag with the existing SLAM launch', 'launch tiago-bag-slam -- bag_path:=/bags/example')]),
    'baseline': ('Run or prepare comparison methods using their existing tools.',
                 [('Run a comparison method on a recording', 'baseline run clio /data/recording /data/clio-results'),
                  ('See baseline run options', 'baseline run --help')]),
    'baseline run': ('Run a comparison method on a recording. Save its results in OUTPUT.',
                     [('Run CLIO', 'baseline run clio /data/recording /data/clio-results'),
                      ('Run HOV-SG', 'baseline run hovsg /data/recording /data/hovsg-results')]),
    'baseline acquire': ('Collect input data using the existing baseline acquisition tool.',
                         [('Show the acquisition tool options', 'baseline acquire -- --help')]),
    'baseline pair': ('Compare two runs using the existing baseline pairing tool.',
                      [('Show the pairing tool options', 'baseline pair -- --help')]),
    'baseline remote': ('Run the existing baseline tool on a remote machine.',
                        [('Show the remote tool options', 'baseline remote -- --help')]),
    'cloud': ('Deploy or run the existing perception service on Modal.',
              [('Deploy the service', 'cloud deploy'), ('Run the service', 'cloud run')]),
    'tiago': ('Manage private PAL Docker and the physical robot network.\n'
              'For new pipeline runs, use run tiago physical or run tiago bag.',
              [('Check the private container', 'tiago check tiago-127-dev'),
               ('Open a container shell', 'tiago shell tiago-127-dev'),
               ('Check the physical robot network settings', 'tiago network status')]),
    'maps': ('List the saved RTAB-Map databases in the workspace.',
             [('List saved maps', 'maps list')]),
}


def configure_help(parser, path=''):
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    key = 'run tiago bag' if path == 'run bag' else path
    if key in COMMAND_HELP:
        description, examples = COMMAND_HELP[key]
        if path == 'run bag':
            description = 'Older spelling of run tiago bag. Prefer run tiago bag.\n' + description
        parser.description = description
        parser.epilog = 'Examples:\n' + '\n\n'.join(
            '  ' + example[0] + ':\n' + '\n'.join('    ./graphapi ' + command for command in example[1:])
            for example in examples)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                configure_help(child, (path + ' ' + name).strip())
            for choice in action._choices_actions:
                child = action.choices[choice.dest]
                if child.description and choice.help != argparse.SUPPRESS:
                    choice.help = child.description.split('\n')[0]


def selected_help(parser, args):
    """Find command help when trailing tool arguments captured --help."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            child = action.choices.get(getattr(args, action.dest, None))
            if child is not None:
                return selected_help(child, args)
    return parser.format_help()

# GRAPH API
Per far partire tutto, in 2 diversi terminali: 

1 conda activate *nome ambiente habitat*
2 ros2 run lost3dsg habitat_camera_node.py se volete quello statico
Oppure
2b cd lost3dgs/src/perception_module
   python3 habitat_camera_objects_node.py se volete quello in cui potete muovere gli oggetti/far partire gli script

Per generare uno script python3 scene_script.py --request "richiesta" --object-scale 1.0 o 0.5 o qualsiasi scala vogliate
Per runnare lo script generato python3 run_habitat_script.py scripts/compiled_script.json

ros2 launch lost3dsg habitat_launch.py

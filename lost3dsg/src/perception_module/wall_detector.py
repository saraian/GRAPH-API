#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
import math
import json

class WallDetector(Node):
    def __init__(self):
        super().__init__('wall_detector')
        # Ascolta il laser con la QoS giusta per la simulazione
        self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, qos_profile_sensor_data)
        # Pubblica i segmenti per il tuo object_manager_3.py
        self.pub = self.create_publisher(String, '/detected_wall_segments', 10)
        self.get_logger().info("Wall Detector avviato! In attesa di /scan_raw...")

    def scan_callback(self, msg):
        walls = []
        current_wall = []
        
        # Distanza massima tra due punti laser per considerarli "lo stesso muro" (in metri)
        TOLERANCE = 0.3 
        # Numero minimo di punti laser vicini per formare un muro reale (filtra il rumore)
        MIN_POINTS = 5

        for i, r in enumerate(msg.ranges):
            # Ignora i punti che il laser non ha colpito (inf, NaN, o fuori range)
            if math.isinf(r) or math.isnan(r) or r < msg.range_min or r > msg.range_max:
                continue
            
            # Calcola le coordinate X, Y del punto rispetto al laser
            angle = msg.angle_min + i * msg.angle_increment
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            
            if not current_wall:
                current_wall.append((x, y))
            else:
                last_x, last_y = current_wall[-1]
                dist = math.hypot(x - last_x, y - last_y)
                
                if dist < TOLERANCE:
                    # Il punto è vicino al precedente, fa parte dello stesso muro
                    current_wall.append((x, y))
                else:
                    # Il punto è lontano, il muro è finito
                    if len(current_wall) >= MIN_POINTS:
                        # Salva [x_inizio, y_inizio, x_fine, y_fine]
                        walls.append([current_wall[0][0], current_wall[0][1], current_wall[-1][0], current_wall[-1][1]])
                    # Inizia un nuovo muro con il punto corrente
                    current_wall = [(x, y)]
        
        # Controlla l'ultimo muro rimasto in canna a fine ciclo
        if len(current_wall) >= MIN_POINTS:
            walls.append([current_wall[0][0], current_wall[0][1], current_wall[-1][0], current_wall[-1][1]])

        # Pubblica il risultato come stringa JSON, come si aspetta l'object manager
        msg_out = String()
        msg_out.data = json.dumps(walls)
        self.pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = WallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
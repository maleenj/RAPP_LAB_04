from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vam_viz_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'web'), glob('web/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maleen',
    maintainer_email='maleen@rapplab.org',
    description='Config-driven WebSocket bridge streaming ROS2 topics to Unity/Unreal.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'viz_bridge_node = vam_viz_bridge.viz_bridge_node:main',
            'test_client = vam_viz_bridge.test_client:main',
            'vote_robot_bridge = vam_viz_bridge.vote_robot_bridge:main',
        ],
    },
)

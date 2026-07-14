from setuptools import setup
import os
from glob import glob

package_name = 'ur_trajectory_publisher'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='UR10 trajectory publisher with board',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_publisher = ur_trajectory_publisher.trajectory_publisher:main',
            'joint_state_capture = ur_trajectory_publisher.joint_state_capture:main'  # Add this line
        ],
    },
)
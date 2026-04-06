from setuptools import setup

package_name = 'tm_teach_playback'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='Maleen Jayasuriya',
    maintainer_email='maleen@todo.todo',
    description='Record and playback joint waypoints for TM12S arm',
    license='MIT',
    entry_points={
        'console_scripts': [
            'record_waypoints = tm_teach_playback.record_waypoints:main',
            'play_waypoints = tm_teach_playback.play_waypoints:main',
        ],
    },
)

import os
import time
#import JSON-RPC Pulse Streamer wrapper class, to use Google-RPC import from pulsestreamer.grpc
from pulsestreamer import PulseStreamer, Sequence, OutputState, findPulseStreamers
# impofr timetagger 
os.environ['TIMETAGGER_INSTALL_PATH'] = 'C:\Program Files\Swabian Instruments\Time Tagger'
import TimeTagger as tt

# activate tagger
_tagger = tt.createTimeTagger()
_tagger.reset()

# set trigger level to 1.5V
_tagger.setTriggerLevel(1, 1.5)
_tagger.sync()

# activate pulser
devices = findPulseStreamers()
# DHCP is activated in factory settings
if devices !=[]:
    ip = devices[0][0]
else:
    # if discovery failed try to connect by the default hostname
    # IP address of the pulse streamer (default hostname is 'pulsestreamer')
    print("No Pulse Streamer found")

#connect to the pulse streamer
pulser = PulseStreamer(ip)

# Print serial number and FPGA-ID
print('Serial: ' + pulser.getSerial())
print('FPGA ID: ' + pulser.getFPGAID())

'''
Set pulser and tagger
'''
ch_sig = 0
ch_gate = 1

click_channel = ch_sig
start_channel = ch_gate
next_channel = -ch_gate

HIGH=1
LOW=0
seq_gate=[]
seq_sig=[]

#define pulse patterns for each channels
# simply add more pulses with ', (time, HIGH/LOW)'
for i in range(1,11):
    seq_gate += [(1500,HIGH),(500,LOW)]
    seq_sig += [(100,LOW)] + i*[(50,HIGH),(50,LOW)] + [(2000-100-i*100, LOW)]
#create the sequence
seq = Sequence()

#set digital channels
seq.setDigital(ch_gate, seq_gate)
seq.setDigital(ch_sig, seq_sig)


seq.plot()

''''
Configer Timetagger
'''
# self._number_of_gates = number_of_gates
bin_width = 10
record_length = 1500
n_bins = 150
n_histograms = 10
assert record_length % bin_width == 0, 'Bins number error!'



pulsed = tt.TimeDifferences(
    tagger=_tagger,
    click_channel=click_channel,
    start_channel=start_channel,
    next_channel=next_channel,
    sync_channel=tt.CHANNEL_UNUSED,
    binwidth=bin_width * 1000, # In ps
    n_bins=int(n_bins),
    n_histograms=n_histograms
)
pulsed.setMaxCounts(1)
time.sleep(0.5)

final = OutputState.ZERO()
pulser.stream(seq, 1, final)


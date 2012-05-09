import numpy as np
import os

import thorns as th
import dsam

from auditory_periphery import AuditoryPeriphery, par_dir



class LopezPoveda2006(AuditoryPeriphery):
    """
    Sumner2002 auditory periphery model with Lopez-Poveda2006 IHCRP model.
    """
    def __init__(self, hsr=100, msr=100, lsr=100, freq=1000.0, animal='gp'):

        self.hsr = hsr
        self.msr = msr
        self.lsr = lsr
        self.animal = animal

        # Outer/middle ear filter
        if self.animal == 'gp':
            self.outer_middle_ear = dsam.EarModule("Filt_MultiBPass")
            self.outer_middle_ear.read_pars(par_dir("filt_GP_A.par"))

            self.outer_middle_ear_B = dsam.EarModule("Filt_MultiBPass")
            self.outer_middle_ear_B.read_pars(par_dir("filt_GP_B.par"))
            dsam.connect(self.outer_middle_ear, self.outer_middle_ear_B)
        elif self.animal == 'human':
            self.outer_middle_ear = dsam.EarModule("Filt_MultiBPass")
            self.outer_middle_ear.read_pars(par_dir("filt_Human.par"))
        else:
            assert False


        # Stapes velocity [Pa -> m/s]
        self.stapes_velocity = dsam.EarModule("Util_mathOp")
        if self.animal == 'gp':
            self.stapes_velocity.set_par("OPERATOR", "SCALE")
            self.stapes_velocity.set_par("OPERAND", 1.7e-9)
            dsam.connect(self.outer_middle_ear_B, self.stapes_velocity)
        elif self.animal == 'human':
            self.stapes_velocity.set_par("OPERATOR", "SCALE")
            self.stapes_velocity.set_par("OPERAND", 1.7e-11)
            dsam.connect(self.outer_middle_ear, self.stapes_velocity)
        else:
            assert False


        # Basilar membrane
        if self.animal == 'gp':
            self.bm = dsam.EarModule("BM_DRNL")
            self.bm.read_pars(par_dir("bm_drnl_gp.par"))
            self.set_freq(freq)
            dsam.connect(self.stapes_velocity, self.bm)
        elif self.animal == 'human':
            self.bm = dsam.EarModule("BM_DRNL")
            self.bm.read_pars(par_dir("drnl_human_Lopez-Poveda2001.par"))
            self.set_freq(freq)
            dsam.connect(self.stapes_velocity, self.bm)
        else:
            assert False


        # IHC receptor potential
        self.ihcrp = dsam.EarModule("IHCRP_LopezPoveda")
        self.ihcrp.read_pars(par_dir("ihcrp_LopezPoveda2006.par"))
        dsam.connect(self.bm, self.ihcrp)

        if self.hsr != 0:
            self.ihc_hsr = dsam.EarModule("IHC_Meddis2000")
            self.ihc_hsr.read_pars(par_dir("ihc_hsr_for_LopezPoveda2006.par"))
            dsam.connect(self.ihcrp, self.ihc_hsr)

            self.anf_hsr = dsam.EarModule("An_SG_Binomial")
            self.anf_hsr.read_pars(par_dir("anf_binomial.par"))
            self.anf_hsr.set_par("NUM_FIBRES", hsr)
            dsam.connect(self.ihc_hsr, self.anf_hsr)

        if self.msr != 0:
            self.ihc_msr = dsam.EarModule("IHC_Meddis2000")
            self.ihc_msr.read_pars(par_dir("ihc_msr_Meddis2002.par"))
            dsam.connect(self.ihcrp, self.ihc_msr)

            self.anf_msr = dsam.EarModule("An_SG_Binomial")
            self.anf_msr.read_pars(par_dir("anf_binomial.par"))
            self.anf_msr.set_par("NUM_FIBRES", msr)
            dsam.connect(self.ihc_msr, self.anf_msr)


        if self.lsr != 0:
            self.ihc_lsr = dsam.EarModule("IHC_Meddis2000")
            self.ihc_lsr.read_pars(par_dir("ihc_lsr_Meddis2002.par"))
            dsam.connect(self.ihcrp, self.ihc_lsr)

            self.anf_lsr = dsam.EarModule("An_SG_Binomial")
            self.anf_lsr.read_pars(par_dir("anf_binomial.par"))
            self.anf_lsr.set_par("NUM_FIBRES", lsr)
            dsam.connect(self.ihc_lsr, self.anf_lsr)



    def set_freq(self, freq):
        # Test for `freq' type, should be either float or tuple
        # flaot: single BM frequency
        # tuple: (min_freq, max_freq, freq_num)
        if isinstance(freq, int):
            freq = float(freq)
        assert (isinstance(freq, tuple) or
                isinstance(freq, float))

        if isinstance(freq, float):
            self.bm.set_par("CF_MODE", "single")
            self.bm.set_par("SINGLE_CF", freq)
        elif isinstance(freq, tuple):
            if self.animal == 'gp':
                self.bm.set_par("CF_MODE", "guinea_pig")
            elif self.animal == 'human':
                self.bm.set_par("CF_MODE", "human")
            else:
                assert False
            self.bm.set_par("MIN_CF", freq[0])
            self.bm.set_par("MAX_CF", freq[1])
            self.bm.set_par("CHANNELS", freq[2])



    def _run_anf(self, anf, fs, times, output_format):
        """
        Run spike generator several times and format the output.
        """
        anf.set_par("PULSE_DURATION", 1.1/fs)

        if output_format == 'spikes':
            anf_db = []
            for run_idx in range(times):
                anf.run()
                anf_signal = anf.get_signal()
                anf_spikes = th.signal_to_spikes(fs, anf_signal)

                for freq_idx,each_freq in enumerate(anf_spikes):
                    anf_db.append( (freq_idx, run_idx, each_freq) )

            anf_output = np.array(anf_db, dtype=[ ('freq', int),
                                                  ('trial', int),
                                                  ('spikes', np.ndarray) ])
        elif output_format == 'signals':
            anf.run()
            anf_output = anf.get_signal()
        else:
            assert False


        return anf_output



    def run(self, fs, sound, times=1, output_format='spikes'):
        """
        Run auditory periphery model.

        sound: audio signal
        fs: sampling frequency
        times: how many many trials
        output_format: format of the output 'spikes' (for spiking
        times), 'signals' (for time function)
        """
        if output_format == 'signals':
            assert times == 1

        fs = float(fs)
        input_module = dsam.EarModule(fs, sound)

        dsam.connect(input_module, self.outer_middle_ear)

        if self.animal == 'gp':
            self.outer_middle_ear.run()
            self.outer_middle_ear_B.run()
        elif self.animal == 'human':
            self.outer_middle_ear.run()
        else:
            self.outer_middle_ear.run()

        dsam.disconnect(input_module, self.outer_middle_ear)

        self.stapes_velocity.run()
        self.bm.run()
        self.ihcrp.run()

        if self.hsr > 0:
            self.ihc_hsr.run()
            hsr_db = self._run_anf(self.anf_hsr, fs, times, output_format)
        else:
            hsr_db = None

        if self.msr > 0:
            self.ihc_msr.run()
            msr_db = self._run_anf(self.anf_msr, fs, times, output_format)
        else:
            msr_db = None

        if self.lsr > 0:
            self.ihc_lsr.run()
            lsr_db = self._run_anf(self.anf_lsr, fs, times, output_format)
        else:
            lsr_db = None


        return hsr_db, msr_db, lsr_db

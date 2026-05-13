function saveUserProfile(results, bestFitness, patientId, patientName, biometrics, outputPath)
% saveUserProfile - Saves the full per-movement GA output as a single profile JSON
%
% Takes the best chromosomes for all 3 movements and writes them to one
% human-readable JSON file with all 70 genes labelled per movement.
%
% INPUT:
%   results      - struct with fields move1, move2, move3, each containing:
%                    .chromosome  - 1x70 vector from runGA
%                    .fitness     - final fitness score from runGA
%   bestFitness  - unused placeholder (kept for legacy compatibility)
%   patientId    - integer patient ID (e.g. 1)
%   patientName  - string patient name (e.g. 'John Doe')
%   biometrics   - struct with fields: upper_arm_length_mm, forearm_length_mm,
%                  shoulder_width_mm
%   outputPath   - full path to write the profile JSON
%
% OUTPUT:
%   Writes the profile JSON to outputPath
%
% Author: Aiden (GA / Fitness Function)
% Project: LIMB - Personalised Motor Rehabilitation

% =========================================================================
% Definitions — must match chromosome gene order exactly
% =========================================================================

jointNames = {
    'sh_flex', 'sh_abd', 'el_flex', 'wr_pron', ...
    'thumb_MCP', 'thumb_PIP', ...
    'index_MCP', 'index_PIP', ...
    'middle_MCP', 'middle_PIP', ...
    'ring_MCP', 'ring_PIP', ...
    'pinky_MCP', 'pinky_PIP'
};

featureNames  = {'max_angle', 'start_angle', 'smoothness', 'time_to_peak_min', 'time_to_peak_max'};
numJoints     = 14;
genesPerJoint = 5;
movements     = {'move1', 'move2', 'move3'};

% =========================================================================
% Build JSON string
% =========================================================================

lines = {};
lines{end+1} = '{';
lines{end+1} = sprintf('  "patient_id": %d,', patientId);
lines{end+1} = sprintf('  "patient_name": "%s",', patientName);
lines{end+1} = sprintf('  "date_recorded": "%s",', datestr(now, 'yyyy-mm-dd'));

% --- Biometrics block ---
lines{end+1} = '  "biometrics": {';
bioFields = fieldnames(biometrics);
for b = 1:length(bioFields)
    fname = bioFields{b};
    val   = biometrics.(fname);
    if b < length(bioFields)
        comma = ',';
    else
        comma = '';
    end
    if isempty(val)
        lines{end+1} = sprintf('    "%s": null%s', fname, comma);
    else
        lines{end+1} = sprintf('    "%s": %.2f%s', fname, val, comma);
    end
end
lines{end+1} = '  },';
% --- Movements block ---
lines{end+1} = '  "movements": {';

for m = 1:length(movements)
    movId      = movements{m};
    movData    = results.(movId);
    chromosome = movData.chromosome;
    fitness    = movData.fitness;

    if m < length(movements)
        movSuffix = ',';
    else
        movSuffix = '';
    end

    lines{end+1} = sprintf('    "%s": {', movId);
    lines{end+1} = sprintf('      "ga_fitness": %.6f,', fitness);
    lines{end+1} = '      "profile": {';

    for j = 1:numJoints
        geneBase = (j - 1) * genesPerJoint;
        jName    = jointNames{j};

        lines{end+1} = sprintf('        "%s": {', jName);

        for f = 1:genesPerJoint
            gIdx  = geneBase + f;
            val   = chromosome(gIdx);
            fname = featureNames{f};

            if f < genesPerJoint
                lines{end+1} = sprintf('          "%s": %.4f,', fname, val);
            else
                lines{end+1} = sprintf('          "%s": %.4f', fname, val);
            end
        end

        if j < numJoints
            lines{end+1} = '        },';
        else
            lines{end+1} = '        }';
        end
    end

    lines{end+1} = '      }';  % close profile

    if m < length(movements)
        lines{end+1} = '    },';
    else
        lines{end+1} = '    }';
    end
end

lines{end+1} = '  }';   % close movements
lines{end+1} = '}';

% =========================================================================
% Write to file
% =========================================================================

fid = fopen(outputPath, 'w');
if fid == -1
    error('saveUserProfile: could not open file for writing: %s', outputPath);
end

for i = 1:length(lines)
    fprintf(fid, '%s\n', lines{i});
end
fclose(fid);

fprintf('[saveUserProfile] Profile saved to: %s\n', outputPath);

end

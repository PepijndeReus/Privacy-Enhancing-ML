import java.nio.charset.StandardCharsets;

import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.ARXConfiguration.AnonymizationAlgorithm;

import org.deidentifier.arx.criteria.KAnonymity;

import java.io.IOException;
import java.io.FileNotFoundException;

import java.io.File;
import java.util.Scanner; 

// how to run the file
// javac -cp .:libraries/* k_anonymity.java
// java k_anonymity k file.yaml

public class k_anonymity {

    public static void main(String[] args) throws IOException, FileNotFoundException {
		
		int k = Integer.parseInt(args[0]);
		String input_file = args[1];
		double suppression = Double.parseDouble(args[2]);
		String type = args[3];

		Data data = Data.create(input_file, StandardCharsets.UTF_8, ',');

		File myobject = new File("hierarchy.txt");
		// File myobject = new File("hierarchy_" + type + ".txt");
        Scanner myReader = new Scanner(myobject);
        while (myReader.hasNextLine()) {
            String line = myReader.nextLine();
            String[] types = line.split(",");
            
			if (types[1].equals("Insensitive")) {
				data.getDefinition().setAttributeType(types[0], AttributeType.INSENSITIVE_ATTRIBUTE);
			} else if (types[1].equals("Identifying")) {
				data.getDefinition().setAttributeType(types[0], AttributeType.IDENTIFYING_ATTRIBUTE);
			} else if (types[1].equals("Quasi_identifying")){
				data.getDefinition().setAttributeType(types[0], AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
			} else if (types[1].equals("Sensitive")) {
				data.getDefinition().setAttributeType(types[0], AttributeType.SENSITIVE_ATTRIBUTE);
			} else {
				data.getDefinition().setAttributeType(types[0], Hierarchy.create(types[1], StandardCharsets.UTF_8, ','));
			}
        }
        myReader.close();

		// Create an instance of the anonymizer
		ARXAnonymizer anonymizer = new ARXAnonymizer();
		ARXConfiguration config = ARXConfiguration.create();
		config.setAlgorithm(AnonymizationAlgorithm.BEST_EFFORT_TOP_DOWN);
		config.addPrivacyModel(new KAnonymity(k));
		config.setSuppressionLimit(suppression);

		// Save files of the anyonymisation
		ARXResult result = anonymizer.anonymize(data, config);
		File f = new File(type + "_" + k + ".csv");
		
		if(!f.exists() && !f.isDirectory()) { 
			result.getOutput(false).save(type + "_" + k + ".csv");
		}
		
	}
}
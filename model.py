import tensorflow as tf
from network import build_network



def get_model():
        resnet_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
        yolo_network = build_network()

        yolo_network.layers[1].set_weights(resnet_model.layers[2].weights)
        yolo_network.layers[2].set_weights(resnet_model.layers[3].weights)

        yolo_network.layers[5].set_weights(resnet_model.layers[7].weights)
        yolo_network.layers[6].set_weights(resnet_model.layers[8].weights)

        yolo_network.layers[8].set_weights(resnet_model.layers[10].weights)
        yolo_network.layers[9].set_weights(resnet_model.layers[11].weights)

        yolo_network.layers[11].set_weights(resnet_model.layers[13].weights)
        yolo_network.layers[12].set_weights(resnet_model.layers[14].weights)
        yolo_network.layers[13].set_weights(resnet_model.layers[15].weights)
        yolo_network.layers[14].set_weights(resnet_model.layers[16].weights)

        yolo_network.layers[17].set_weights(resnet_model.layers[19].weights)
        yolo_network.layers[18].set_weights(resnet_model.layers[20].weights)

        yolo_network.layers[20].set_weights(resnet_model.layers[22].weights)
        yolo_network.layers[21].set_weights(resnet_model.layers[23].weights)

        yolo_network.layers[23].set_weights(resnet_model.layers[25].weights)
        yolo_network.layers[24].set_weights(resnet_model.layers[26].weights)

        yolo_network.layers[27].set_weights(resnet_model.layers[29].weights)
        yolo_network.layers[28].set_weights(resnet_model.layers[30].weights)

        yolo_network.layers[30].set_weights(resnet_model.layers[32].weights)
        yolo_network.layers[31].set_weights(resnet_model.layers[33].weights)

        yolo_network.layers[33].set_weights(resnet_model.layers[35].weights)
        yolo_network.layers[34].set_weights(resnet_model.layers[36].weights)

        yolo_network.layers[37].set_weights(resnet_model.layers[39].weights)
        yolo_network.layers[38].set_weights(resnet_model.layers[40].weights)

        yolo_network.layers[40].set_weights(resnet_model.layers[42].weights)
        yolo_network.layers[41].set_weights(resnet_model.layers[43].weights)

        yolo_network.layers[44].set_weights(resnet_model.layers[45].weights)
        yolo_network.layers[43].set_weights(resnet_model.layers[46].weights)
        yolo_network.layers[45].set_weights(resnet_model.layers[47].weights)
        yolo_network.layers[46].set_weights(resnet_model.layers[48].weights)

        yolo_network.layers[49].set_weights(resnet_model.layers[51].weights)
        yolo_network.layers[50].set_weights(resnet_model.layers[52].weights)

        yolo_network.layers[52].set_weights(resnet_model.layers[54].weights)
        yolo_network.layers[53].set_weights(resnet_model.layers[55].weights)

        yolo_network.layers[55].set_weights(resnet_model.layers[57].weights)
        yolo_network.layers[56].set_weights(resnet_model.layers[58].weights)

        yolo_network.layers[59].set_weights(resnet_model.layers[61].weights)
        yolo_network.layers[60].set_weights(resnet_model.layers[62].weights)

        yolo_network.layers[62].set_weights(resnet_model.layers[64].weights)
        yolo_network.layers[63].set_weights(resnet_model.layers[65].weights)

        yolo_network.layers[65].set_weights(resnet_model.layers[67].weights)
        yolo_network.layers[66].set_weights(resnet_model.layers[68].weights)

        yolo_network.layers[69].set_weights(resnet_model.layers[71].weights)
        yolo_network.layers[70].set_weights(resnet_model.layers[72].weights)

        yolo_network.layers[72].set_weights(resnet_model.layers[74].weights)
        yolo_network.layers[73].set_weights(resnet_model.layers[75].weights)

        yolo_network.layers[75].set_weights(resnet_model.layers[77].weights)
        yolo_network.layers[76].set_weights(resnet_model.layers[78].weights)

        yolo_network.layers[79].set_weights(resnet_model.layers[81].weights)
        yolo_network.layers[80].set_weights(resnet_model.layers[82].weights)

        yolo_network.layers[82].set_weights(resnet_model.layers[84].weights)
        yolo_network.layers[83].set_weights(resnet_model.layers[85].weights)

        yolo_network.layers[86].set_weights(resnet_model.layers[87].weights)
        yolo_network.layers[85].set_weights(resnet_model.layers[88].weights)
        yolo_network.layers[87].set_weights(resnet_model.layers[89].weights)
        yolo_network.layers[88].set_weights(resnet_model.layers[90].weights)

        yolo_network.layers[91].set_weights(resnet_model.layers[93].weights)
        yolo_network.layers[92].set_weights(resnet_model.layers[94].weights)

        yolo_network.layers[94].set_weights(resnet_model.layers[96].weights)
        yolo_network.layers[95].set_weights(resnet_model.layers[97].weights)

        yolo_network.layers[97].set_weights(resnet_model.layers[99].weights)
        yolo_network.layers[98].set_weights(resnet_model.layers[100].weights)

        yolo_network.layers[101].set_weights(resnet_model.layers[103].weights)
        yolo_network.layers[102].set_weights(resnet_model.layers[104].weights)

        yolo_network.layers[104].set_weights(resnet_model.layers[106].weights)
        yolo_network.layers[105].set_weights(resnet_model.layers[107].weights)

        yolo_network.layers[107].set_weights(resnet_model.layers[109].weights)
        yolo_network.layers[108].set_weights(resnet_model.layers[110].weights)

        yolo_network.layers[111].set_weights(resnet_model.layers[113].weights)
        yolo_network.layers[112].set_weights(resnet_model.layers[114].weights)

        yolo_network.layers[114].set_weights(resnet_model.layers[116].weights)
        yolo_network.layers[115].set_weights(resnet_model.layers[117].weights)

        yolo_network.layers[117].set_weights(resnet_model.layers[119].weights)
        yolo_network.layers[118].set_weights(resnet_model.layers[120].weights)

        yolo_network.layers[121].set_weights(resnet_model.layers[123].weights)
        yolo_network.layers[122].set_weights(resnet_model.layers[124].weights)

        yolo_network.layers[124].set_weights(resnet_model.layers[126].weights)
        yolo_network.layers[125].set_weights(resnet_model.layers[127].weights)

        yolo_network.layers[127].set_weights(resnet_model.layers[129].weights)
        yolo_network.layers[128].set_weights(resnet_model.layers[130].weights)

        yolo_network.layers[131].set_weights(resnet_model.layers[133].weights)
        yolo_network.layers[132].set_weights(resnet_model.layers[134].weights)

        yolo_network.layers[134].set_weights(resnet_model.layers[136].weights)
        yolo_network.layers[135].set_weights(resnet_model.layers[137].weights)

        yolo_network.layers[137].set_weights(resnet_model.layers[139].weights)
        yolo_network.layers[138].set_weights(resnet_model.layers[140].weights)
        
        return yolo_network


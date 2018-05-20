import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AppComponent } from './components/app.component';
import { HeaderComponent } from './components/header/header.component';
import { ClassificationComponent } from './components/classification/classification.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    ClassificationComponent
  ],
  imports: [
    BrowserModule,
    FormsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
